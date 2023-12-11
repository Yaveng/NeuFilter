import time
import sys
from dataloader import *
from model import *
from sklearn.metrics import roc_auc_score


def eval_epoch(args, mode='valid'):
    args.datapath = "data/{}.csv".format(args.network)
    device = 'cuda:{}'.format(args.gpu)

    # load dataset
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,\
     item2id, item_sequence_id, item_timediffs_sequence,timestamp_sequence, feature_sequence,y_true] = load_network(args)
    num_interactions = len(user_sequence_id)
    num_features = len(feature_sequence[0])
    num_users = len(user2id)
    num_items = len(item2id) + 1
    true_labels_ratio = len(y_true)/(sum(y_true)+1)

    train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
    test_start_idx = int(num_interactions * (args.train_proportion + 0.2))
    test_end_idx = int(num_interactions * (args.train_proportion + 0.4))
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / 500

    model = NeuFilter(args, num_features, num_users, num_items, device).to(device)
    weight = torch.Tensor([1,true_labels_ratio]).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch, device)

    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)
    item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
    item_embeddings = item_embeddings.clone()
    item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
    item_embeddings_static = item_embeddings_static.clone()
    user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
    user_embeddings = user_embeddings.clone()
    user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
    user_embeddings_static = user_embeddings_static.clone()

    validation_predicted_y = []
    test_predicted_y = []
    validation_true_y = []
    test_true_y = []
    tbatch_start_time = None
    loss = 0
    t_start = time.time()
    end_idx = test_start_idx if mode == 'valid' else test_end_idx
    for j in range(train_end_idx, end_idx):
        # Load  j-th interaction
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        # Load user and item embedding
        user_embedding_input = user_embeddings[torch.LongTensor([userid]).to(device)]
        user_embedding_static_input = user_embeddings_static[torch.LongTensor([userid]).to(device)]
        item_embedding_input = item_embeddings[torch.LongTensor([itemid]).to(device)]
        item_embedding_static_input = item_embeddings_static[torch.LongTensor([itemid]).to(device)]
        feature_tensor = Variable(torch.Tensor(feature).to(device)).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).to(device)).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).to(device)).unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.LongTensor([itemid_previous]).to(device)]

        # Predict user and item embedding
        user_projected_embedding, item_projected_embedding = model.forward(user_embedding_input, item_embedding_input, user_timediffs=user_timediffs_tensor, item_timediffs=item_timediffs_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.LongTensor([itemid_previous]).to(device)], user_embedding_static_input], dim=1)

        # Predict embedding of item that user will interact with
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # Calculate prediction loss
        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # Update user and item embedding
        user_embedding_output, user_emb_reg = model.forward(user_embedding_input, item_embedding_input, user_prior=user_projected_embedding, users=[userid], features=feature_tensor, select='user_update')
        item_embedding_output, item_emb_reg = model.forward(user_embedding_input, item_embedding_input, item_prior=item_projected_embedding, items=[itemid], features=feature_tensor, select='item_update')

        # Save embeddings
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0)
        user_embeddings[userid,:] = user_embedding_output.squeeze(0)
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # Calculate regularization terms
        loss += args.reg_factor1 * MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += args.reg_factor1 * MSELoss(user_embedding_output, user_embedding_input.detach())
        loss += args.reg_factor2 * (user_emb_reg + item_emb_reg)

        # Calculate state change loss
        loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss, device)

        # update model parameters
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            t_end = time.time()
            print('[{}] Processing {}-th/{}-th interactions with loss {:.2f} elapses {:.2f} min'
                  .format('Valid' if mode == 'valid' else 'Test', j, end_idx, loss, (t_end - t_start) / 60.0))

            # Reset loss for next t-batch
            loss = 0
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()
            model.user_kf.h.detach_()
            model.item_kf.h.detach_()

        prob = model.predict_label(user_embedding_output)

        if j < test_start_idx:
            validation_predicted_y.extend(prob.data.cpu().numpy())
            validation_true_y.extend([y_true[j]])
        else:
            test_predicted_y.extend(prob.data.cpu().numpy())
            test_true_y.extend([y_true[j]])
    if not os.path.exists('results'):
        os.mkdir('results')
    output_fname = "results/state_change_prediction_{}.txt".format(args.postfix)
    if mode == 'valid':
        validation_predicted_y = np.array(validation_predicted_y)
        performance_results = []
        auc = roc_auc_score(validation_true_y, validation_predicted_y[:, 1])
        performance_results.append(auc)
        performance_results.append(args.epoch)

        fw = open(output_fname, "a+")
        fw.write('[Valid {}]  '.format(args.epoch))
        fw.write('AUC:{:.3f}'.format(auc) + "\n")
        fw.flush()
        fw.close()
        return performance_results
    else:
        validation_predicted_y = np.array(validation_predicted_y)
        test_predicted_y = np.array(test_predicted_y)
        valid_auc = roc_auc_score(validation_true_y, validation_predicted_y[:,1])
        test_auc = roc_auc_score(test_true_y, test_predicted_y[:,1])
        fw = open(output_fname, "a+")
        fw.write('[Test {}]  '.format(args.epoch))
        fw.write('AUC(Valid):{:.3f}\tAUC(Test):{:.3f}'.format(valid_auc, test_auc)+ "\n")
        fw.flush()
        fw.close()


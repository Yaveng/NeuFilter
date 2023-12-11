import time
import sys
from dataloader import *
import model as lib
from model import *
import torch.optim as optim


def train(args):
    args.datapath = "data/{}.csv".format(args.network)
    device = 'cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu'

    # Load dataset
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence, timestamp_sequence, feature_sequence, y_true] = load_network(args)
    num_interactions = len(user_sequence_id)
    num_users = len(user2id)
    num_items = len(item2id) + 1
    num_features = len(feature_sequence[0])
    true_labels_ratio = len(y_true)/(1.0+sum(y_true))
    train_end_idx = int(num_interactions * args.train_proportion)
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / 500

    # Initialize model
    model = NeuFilter(args, num_features, num_users, num_items, device).to(device)
    weight = torch.Tensor([1,true_labels_ratio]).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()

    initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(device), dim=0))
    initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(device), dim=0))
    model.initial_user_embedding = initial_user_embedding
    model.initial_item_embedding = initial_item_embedding

    user_embeddings = initial_user_embedding.repeat(num_users, 1)  # initialize all users to the same embedding
    item_embeddings = initial_item_embedding.repeat(num_items, 1)  # initialize all items to the same embedding
    item_embedding_static = Variable(torch.eye(num_items).to(device))  # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(num_users).to(device))  # one-hot vectors for static embeddings

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    is_first_epoch = True
    cached_tbatches_user = {}
    cached_tbatches_item = {}
    cached_tbatches_interactionids = {}
    cached_tbatches_feature = {}
    cached_tbatches_user_timediffs = {}
    cached_tbatches_item_timediffs = {}
    cached_tbatches_previous_item = {}

    t_start = time.time()
    for ep in range(args.epochs):
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(device))
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(device))

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None

        # Train till the end of training interaction idx
        for j in range(train_end_idx):
            if is_first_epoch:
                # Load j-th interaction
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                # Create t-batch
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            if timestamp - tbatch_start_time > tbatch_timespan:
                # Reset start time for the next t-batch
                tbatch_start_time = timestamp
                if not is_first_epoch:
                    lib.current_tbatches_user = cached_tbatches_user[timestamp]
                    lib.current_tbatches_item = cached_tbatches_item[timestamp]
                    lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                    lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                for i in range(len(lib.current_tbatches_user)):
                    total_interaction_count += len(lib.current_tbatches_interactionids[i])

                    # Load the current t-batch
                    if is_first_epoch:
                        lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).to(device)
                        lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).to(device)
                        lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).to(device)
                        lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).to(device)

                        lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device)
                        lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).to(device)
                        lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).to(device)

                    tbatch_userids = lib.current_tbatches_user[i]
                    tbatch_itemids = lib.current_tbatches_item[i]
                    tbatch_interactionids = lib.current_tbatches_interactionids[i]
                    feature_tensor = Variable(lib.current_tbatches_feature[i])
                    user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                    item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                    tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                    # Predict user and item embedding
                    user_embedding_input, item_embedding_input = user_embeddings[tbatch_userids,:], item_embeddings[tbatch_itemids,:]
                    user_projected_embedding, item_projected_embedding = model.forward(user_embedding_input, item_embedding_input, user_timediffs=user_timediffs_tensor, item_timediffs=item_timediffs_tensor, select='project')

                    # Predict the embedding of item that user will interact with
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # Calculate the prediction loss
                    loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                    # Update user and item embedding
                    user_embedding_output, user_emb_reg = model.forward(user_embedding_input, item_embedding_input, user_prior=user_projected_embedding, users=tbatch_userids, features=feature_tensor, select='user_update')
                    item_embedding_output, item_emb_reg = model.forward(user_embedding_input, item_embedding_input, item_prior=item_projected_embedding, items=tbatch_itemids, features=feature_tensor, select='item_update')

                    item_embeddings[tbatch_itemids,:] = item_embedding_output
                    user_embeddings[tbatch_userids,:] = user_embedding_output

                    user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                    item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                    # Calculate regularization terms
                    loss += args.reg_factor1 * MSELoss(item_embedding_output, item_embedding_input.detach())
                    loss += args.reg_factor1 * MSELoss(user_embedding_output, user_embedding_input.detach())
                    loss += args.reg_factor2 * (user_emb_reg + item_emb_reg)

                    # Calculate state change loss
                    loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss, device)

                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Reset loss for next t-batch
                loss = 0
                item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_()
                user_embeddings_timeseries.detach_()
                model.user_kf.h.detach_()
                model.item_kf.h.detach_()

                # Reinitialize
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.current_tbatches_user
                    cached_tbatches_item[timestamp] = lib.current_tbatches_item
                    cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                    cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item

                    reinitialize_tbatches()
            t_end = time.time()
            print('[Train] Processing {}th/{}th interactions in {}th/{}th epoch elapses {:.2f} min'.format(j, train_end_idx, ep, args.epochs, (t_end-t_start)/60.0))

        is_first_epoch = False
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)


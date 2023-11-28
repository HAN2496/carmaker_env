def pretrain(dataset, n_epochs=10, learning_rate=1e-4,
             adam_epsilon=1e-8, val_interval=None):
    """
    Pretrain a model using behavior cloning:
    supervised learning given an expert dataset.

    NOTE: only Box and Discrete spaces are supported for now.

    :param dataset: (ExpertDataset) Dataset manager
    :param n_epochs: (int) Number of iterations on the training set
    :param learning_rate: (float) Learning rate
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param val_interval: (int) Report training and validation losses every n epochs.
        By default, every 10th of the maximum number of epochs.
    :return: (BaseRLModel) the pretrained model
    """
    continuous_actions = isinstance(self.action_space, gym.spaces.Box)
    discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

    assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

    # Validate the model every 10% of the total number of iteration
    if val_interval is None:
        # Prevent modulo by zero
        if n_epochs < 10:
            val_interval = 1
        else:
            val_interval = int(n_epochs / 10)

        self.sess.run(tf.global_variables_initializer())

    if self.verbose > 0:
        print("Pretraining with Behavior Cloning...")

    for epoch_idx in range(int(n_epochs)):
        train_loss = 0.0
        # Full pass on the training set
        for _ in range(len(dataset.train_loader)):
            expert_obs, expert_actions = dataset.get_next_batch('train')
            feed_dict = {
                obs_ph: expert_obs,
                actions_ph: expert_actions,
            }
            train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
            train_loss += train_loss_

        train_loss /= len(dataset.train_loader)

        if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
            val_loss = 0.0
            # Full pass on the validation set
            for _ in range(len(dataset.val_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('val')
                val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                    actions_ph: expert_actions})
                val_loss += val_loss_

            val_loss /= len(dataset.val_loader)
            if self.verbose > 0:
                print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                print('Epoch {}'.format(epoch_idx + 1))
                print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                print()
        # Free memory
        del expert_obs, expert_actions
    if self.verbose > 0:
        print("Pretraining done.")
    return self

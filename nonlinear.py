import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pandas as pd
import ipdb
from sklearn import svm
from matplotlib import pyplot as plt
import sys
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from multiprocessing import Pool
import random


def f_global_balancing_regression(train_or_test, X_in, Y_in, beta_true, beta_error_pre):
        if train_or_test == 1:
            learning_rate = 0.005
            num_step = 20000
            tol = 1e-8
            tf.reset_default_graph()
            Weight, iterate = f_global_balancing(X_in, learning_rate, num_step, tol)

            return Weight

        else:
            tf.reset_default_graph()
            RMSE, beta_hat = f_weighted_regression(0, X_in, Y_in, np.ones([X_in.shape[0], 1]), 0, 0, 0, 0, 0)

        return RMSE, beta_hat, iterate

def f_global_balancing(X_in, learning_rate, num_steps, tol):
        n, p = X_in.shape

        display_step = n

        X = tf.placeholder("float64", [None, p])
        G = tf.Variable(tf.ones([n, 1], "float64"))
        one = tf.ones([n, p-1, 1], "float64")
        loss_balancing = tf.constant(0, tf.float64)
        loss_balancing2 = tf.constant(0, tf.float64)

        A_tensor = tf.transpose(
            tf.concat([tf.expand_dims((X[:, 1:] * G * G) ** 2, -1), tf.expand_dims((X[:, 1:] * G * G), -1), one], 2), [1, 0, 2])
        y = tf.transpose(tf.tile(tf.expand_dims(tf.expand_dims(X[:, 0], -1) * G * G, -1), multiples=[1, 1, p-1]), [2, 0, 1])
        for i in range(1, p):
            temp_x = tf.concat([X[:, :i], X[:, i+1:]], axis=1)
            temp_y = tf.transpose(tf.tile(tf.expand_dims(tf.expand_dims(X[:, i], -1) * G * G, -1), multiples=[1, 1, p-1]), [2, 0, 1])
            A_tensor = tf.concat([A_tensor, tf.transpose(
            tf.concat([tf.expand_dims((temp_x * G * G) ** 2, -1), tf.expand_dims((temp_x * G * G), -1), one], 2), [1, 0, 2])], axis=0)
            y = tf.concat([y, temp_y], axis=0)
        
        print(y.shape)
        print(A_tensor.shape)
        solution = tf.linalg.lstsq(A_tensor, y, l2_regularizer=0.05, fast=True, name=None)
        loss_balancing = loss_balancing + tf.norm(solution[:, 0, :], ord=2)
        loss_balancing = loss_balancing + tf.norm(solution[:, 1, :], ord=2)
        loss_weight_sum = (tf.reduce_sum(G * G) - n) ** 2
        loss_weight_l2 = tf.reduce_sum((G * G) ** 2)
        loss = lambda1 * loss_balancing + lambda2 * loss_weight_sum + 0.00005 * loss_weight_l2

        LEARNING_RATE_BASE = learning_rate
        LEARNING_RATE_DECAY = 0.99
        LEARNING_RATE_STEP = 100
        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                   gloabl_steps,
                                                   LEARNING_RATE_STEP,
                                                   LEARNING_RATE_DECAY,
                                                   staircase=True)

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, gloabl_steps)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        l_pre = 0
        iterate = 0

        for i in range(1, num_steps + 1):
            iterate = i
            _, l, l_balancing, l_balancing2, l_weight_sum, l_weight_l2 = sess.run(
                [optimizer, loss, loss_balancing, loss_balancing2, loss_weight_sum, loss_weight_l2],
                feed_dict={X: X_in})
            if abs(l - l_pre) <= tol:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f ... %f' % (
                    i, l, l_balancing, l_balancing2, l_weight_sum, l_weight_l2))
                break
            l_pre = l
            if i % 2000 == 0 or i == 1:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f ... %f' % (
                    i, l, l_balancing, l_balancing2, l_weight_sum, l_weight_l2))

        W_final = sess.run(G * G)
        fw = open('weights_global_balancing.txt', 'w')
        for items in W_final:
            fw.write('' + str(items[0]) + '\n')
        fw.close()

        Weight = sess.run([G * G])

        return Weight[0], iterate







def f_weighted_regression(train_or_test, X_in, Y_in, Weight_in, learning_rate, num_steps, tol, beta_true,
                              beta_error_pre):
        n, p = X_in.shape

        display_step = 1000

        X = tf.placeholder("float", [None, p])
        Y = tf.placeholder("float", [None, 1])
        W = tf.placeholder("float", [None, 1])

        beta = tf.Variable(tf.random_normal([p, 1]))
        b = tf.Variable(tf.random_normal([1]))
        hypothesis = tf.matmul(X, beta) + b

        saver = tf.train.Saver()
        sess = tf.Session()

        if train_or_test == 1:
            loss_predictive = tf.divide(tf.reduce_sum(W * (Y - hypothesis) ** 2), tf.reduce_sum(W))
            loss_l1 = tf.reduce_sum(tf.abs(beta))

            loss = 1 * loss_predictive + 0.0 * loss_l1

            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

            sess.run(tf.global_variables_initializer())

            l_pre = 0
            for i in range(1, num_steps + 1):
                _, l, l_predictive, l_l1 = sess.run([optimizer, loss, loss_predictive, loss_l1],
                                                    feed_dict={X: X_in, W: Weight_in, Y: Y_in})
                if abs(l - l_pre) <= tol:
                    print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_predictive, l_l1))
                    break
                l_pre = l
                if i % display_step == 0 or i == 1:
                    print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_predictive, l_l1))

            beta_estimated_error = np.sum(np.abs(sess.run(beta) - beta_true))
            if beta_estimated_error < beta_error_pre:
                if not os.path.isdir('models/f_weighted_regression/'):
                    os.makedirs('models/f_weighted_regression/')
                saver.save(sess, 'models/f_weighted_regression/f_weighted_regression.ckpt')


        else:
            saver.restore(sess, 'models/f_weighted_regression/f_weighted_regression.ckpt')

        RMSE = tf.sqrt(tf.reduce_mean((Y - hypothesis) ** 2))
        RMSE_error, beta_hat = sess.run([RMSE, beta], feed_dict={X: X_in, Y: Y_in})

        return RMSE_error, beta_hat


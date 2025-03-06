#This is a Python Tensorflow 2.18.+ implementation of the Cautious optimizers paper
#Written by Ammar Qammaz, with boilerplate code from ChatGPT!
#For an original implementation see: https://github.com/kyleliang919/C-Optim
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adamw.py#L132-L136

import tensorflow as tf
import keras
from keras.optimizers import Optimizer

"""
Cautious Optimizers: Improving Training with One Line of Code
Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu

    AdamW has been the default optimizer for transformer pretraining. For many years, our community searched for faster and more stable optimizers with only constrained positive outcomes. In this work, we propose a single-line modification in Pytorch to any momentum-based optimizer, which we rename cautious optimizer, e.g. C-AdamW and C-Lion. Our theoretical result shows that this modification preserves Adam's Hamiltonian function and it does not break the convergence guarantee under the Lyapunov analysis. In addition, a whole new family of optimizers is revealed by our theoretical insight. Among them, we pick the simplest one for empirical experiments, showing not only speed-up on Llama and MAE pretraining up to 1.47 times, but also better results in LLM post-training tasks. 

https://arxiv.org/abs/2411.16085
"""

class AdamWCautious(Optimizer):
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 weight_decay=1e-2,
                 amsgrad=False,
                 caution=True,
                 clipnorm=None,
                 clipvalue=None,
                 name="AdamWCautious",
                 **kwargs):
        super().__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1        = beta_1
        self.beta_2        = beta_2
        self.epsilon       = epsilon
        self.weight_decay  = weight_decay
        self.amsgrad       = amsgrad
        self.caution       = caution
        self.clipnorm      = clipnorm
        self.clipvalue     = clipvalue

    def build(self, var_list):
        self.m = {v: tf.Variable(tf.zeros_like(v), trainable=False) for v in var_list}
        self.v = {v: tf.Variable(tf.zeros_like(v), trainable=False) for v in var_list}
        if self.amsgrad:
            self.v_hat = {v: tf.Variable(tf.zeros_like(v), trainable=False) for v in var_list}
        self.t = tf.Variable(0, dtype=tf.int64, trainable=False)
    
    @tf.function
    def apply_gradients(self, grads_and_vars):
        self.t.assign_add(1)
        lr_t = self.learning_rate * tf.sqrt(1.0 - self.beta_2 ** tf.cast(self.t, tf.float32)) / (1.0 - self.beta_1 ** tf.cast(self.t, tf.float32))
        
        for (grad, var) in grads_and_vars:
            if grad is None:
                continue
            
            # Apply gradient clipping
            if self.clipnorm is not None:
                grad = tf.clip_by_norm(grad, self.clipnorm)
            if self.clipvalue is not None:
                grad = tf.clip_by_value(grad, -self.clipvalue, self.clipvalue)
            
            # Apply weight decay
            var.assign_sub(self.learning_rate * self.weight_decay * var)
            
            # Update first and second moment estimates
            self.m[var].assign(self.beta_1 * self.m[var] + (1.0 - self.beta_1) * grad)
            self.v[var].assign(self.beta_2 * self.v[var] + (1.0 - self.beta_2) * tf.square(grad))
            
            if self.amsgrad:
                self.v_hat[var].assign(tf.maximum(self.v_hat[var], self.v[var]))
                denom = tf.sqrt(self.v_hat[var]) + self.epsilon
            else:
                denom = tf.sqrt(self.v[var]) + self.epsilon
            
            step = self.m[var] / denom
            
            if self.caution:
                # Apply cautious update
                mask = tf.cast(self.m[var] * grad > 0, dtype=tf.float32)
                mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)  # Avoid division by zero
                step *= mask
            
            var.assign_sub(lr_t * step)




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    #If this file is run it will make a foo network to test the optimizer
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = AdamWCautious(learning_rate=1e-3, caution=True, clipnorm=1.0)
    
    # Dummy data
    x = tf.random.normal((5, 3))
    y = tf.random.normal((5, 1))
    
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, predictions))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    print("Test completed successfully.")

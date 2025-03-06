#This is a Python Tensorflow 2.18.+ implementation of the Cautious optimizers paper
#Written by Ammar Qammaz, with boilerplate code from ChatGPT!
#For an original implementation see: https://github.com/kyleliang919/C-Optim
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adamw.py#L132-L136

import tensorflow as tf
import keras

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


"""
Cautious Optimizers: Improving Training with One Line of Code
Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu

    AdamW has been the default optimizer for transformer pretraining. For many years, our community searched for faster and more stable optimizers with only constrained positive outcomes. In this work, we propose a single-line modification in Pytorch to any momentum-based optimizer, which we rename cautious optimizer, e.g. C-AdamW and C-Lion. Our theoretical result shows that this modification preserves Adam's Hamiltonian function and it does not break the convergence guarantee under the Lyapunov analysis. In addition, a whole new family of optimizers is revealed by our theoretical insight. Among them, we pick the simplest one for empirical experiments, showing not only speed-up on Llama and MAE pretraining up to 1.47 times, but also better results in LLM post-training tasks. 

https://arxiv.org/abs/2411.16085
"""
 
@keras_export(["keras.optimizers.AdamWCautious"])
class AdamWCautious(optimizer.Optimizer):
    """Optimizer that implements the AdamW algorithm with cautious behavior.

    This optimizer is based on the AdamW algorithm but includes additional
    cautious updates as per the "Cautious Optimizers" paper (https://arxiv.org/abs/2411.16085).

    Args:
        learning_rate: A float, a `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to use. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns
            the actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns
            the actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to `0.999`.
        epsilon: A small constant for numerical stability. Defaults to `1e-7`.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence
            of Adam and beyond". Defaults to `False`.
        weight_decay: Weight decay coefficient. Defaults to `None`.
        caution: Boolean. Whether to apply the cautious behavior to the optimizer. Defaults to `False`.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        caution=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamw_cautious",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.caution = caution

    def build(self, var_list):
        """Initialize optimizer variables (momentums, velocities, and optionally velocity_hats)."""
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(reference_variable=var, name="momentum")
            )
            self._velocities.append(
                self.add_variable_from_reference(reference_variable=var, name="velocity")
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(reference_variable=var, name="velocity_hat")
                )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1))
        self.assign_add(v, ops.multiply(ops.subtract(ops.square(gradient), v), 1 - self.beta_2))

        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat

        if self.caution:
            # Apply caution behavior: mask based on gradient direction and apply cautious update.
            mask = (m * gradient > 0).to(gradient.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            m = m * mask

        self.assign_sub(
            variable,
            ops.divide(ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "caution": self.caution,
            }
        )
        return config

AdamWCautious.__doc__ = AdamWCautious.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import cifar10 

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build a simple CNN model
    model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

    optimizer = AdamWCautious(learning_rate=0.001,clipvalue=1.0)
    # Compile the model with the AdamWCautious optimizer
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    print("Test completed successfully.")

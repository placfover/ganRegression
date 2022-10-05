import keras
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, concatenate


#生成器の構造、入力データによって、異なる活性化関数とネットワークの構造があります
def build_generator(network):
    seed = network.seed
    
    #正規分布に従って重みを初期化します(按照正态分布生成随机张量的初始化器)
    #keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    
    random_normal = keras.initializers.RandomNormal(seed=seed)
    
    #活性化関数によって、異なる初期化方法を使う
    
    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")
       
    
    #合成データの場合、ネットワークの構造
    
    # linear and sinus datasets  
    # This will input x & noise and will output Y.
    if network.architecture == 1:
        
        #合成データとノイズをそれぞ3層のMLPに渡す
        #Input()はKerasテンソルのインスタンス化に使われます
        #Input(shape=None, batch_shape=None,name=None, dtype=None, sparse=False, tensor=None)
        
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise)
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        
        #出力を結合させ、3層のMLPに渡す
        concat = concatenate([x_output, noise_output])
        output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    # heteroscedastic, exp and multi-modal datasets
    elif network.architecture == 2:
        #合成データとノイズをそれぞ3層のMLPに渡す
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise)
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise_output)

        #出力を結合させ、3層のMLPに渡す
        concat = concatenate([x_output, noise_output])
        output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    
    #現実世界のデータの場合、ネットワークの構造
    # CA-housing and ailerons
    elif network.architecture == 3:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    elif network.architecture == 4:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    elif network.architecture == 5:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    else:
        raise NotImplementedError("Architecture does not exist")

    return model


#識別器の構造
def build_discriminator(network):
    seed = network.seed
    
    #一様分布に従って重みを初期化します(按照均匀分布生成随机张量的初始化器)
    #keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    
    random_uniform = keras.initializers.RandomUniform(seed=seed)

    #活性化関数によって、異なる初期化方法を使う
    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    # linear and sinus datasets
    if network.architecture == 1:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    # heteroscedastic, exp and multi-modal datasets
    elif network.architecture == 2:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    # CA-housing and ailerons
    elif network.architecture == 3:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    elif network.architecture == 4:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(25, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    elif network.architecture == 5:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    else:
        raise NotImplementedError("Architecture does not exist")

    return model

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input,multiply, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization,Embedding, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose,Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam
from functools import partial
import keras.backend as K
import sys
from HandleData import read_utk_face
import numpy as np
from PIL import Image
import tensorflow as tf
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        weights = K.random_uniform((32, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class ImprovedWGAN():
    def __init__(self):
	self.GRADIENT_PENALTY_WEIGHT = 10
        self.img_rows = 128 
        self.img_cols = 128
        self.channels = 1
	self.noise_shape = 100
	self.num_classes = 11
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
	self.latent_dim = 100
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = Adam(0.0001, beta_1 = 0.5, beta_2=0.9) 
	#optimizer = RMSprop(lr=0.00005)
	#optimizer = RMSprop(lr=0.00005)
        # Build the generator and discriminator
        self.generator = self.build_generator_exmpl()
        self.discriminator = self.build_discriminator_exmpl()

        #-------------------------------
        # Construct Computational Graph
        #       for Discriminator
        #-------------------------------

        # Freeze generator's layers while training discriminator
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        # Noise input
        z_disc = Input(shape=(100,))
        label_critic = Input(shape=(1,))
	# Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc,label_critic])
	print(fake_img.shape)
        # Discriminator determines validity of the real and fake images
        fake = self.discriminator([fake_img, label_critic])
        real = self.discriminator([real_img, label_critic]) # shadyyyy but I guess 

        # Construct weighted average between real and fake images
        merged_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        valid_merged = self.discriminator([merged_img,label_critic])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=merged_img,
			  gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.discriminator_model = Model(inputs=[real_img, z_disc, label_critic],
                            outputs=[real, fake, valid_merged])
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the discriminator's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        label_gen = Input(shape=(1,))
	# Generate images based of noise
        img = self.generator([z_gen, label_gen])
        # Discriminator determines validity
        valid = self.discriminator([img,label_gen])
        # Defines generator model
        self.generator_model = Model(inputs=[z_gen,label_gen],outputs=[valid])
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples,gradient_penalty_weight):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
	#gradients = K.gradients(K.sum(y_pred),averaged_samples)
	#gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
	#gradient_penalty = gradient_penalty_weight*K.square(1 - gradient_l2_norm)
	#return K.mean(gradient_penalty)
	gradients = K.gradients(y_pred,averaged_samples)[0]
	gradients_sqr = K.square(gradients)
	gradients_sqr_sum = K.sum(gradients_sqr,axis=0)
	gradient_l2_norm = K.sqrt(gradients_sqr_sum)
	gradient_penalty = gradient_penalty_weight * K.square(1- gradient_l2_norm)
	return K.mean(gradient_penalty)
	

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_generator_exmpl(self):
	noise_shape = (100,)
	dim = int(self.img_rows / 4) 
	depth = 64
	model = Sequential()
	#model.add(Dense(depth,input_shape=noise_shape))
	#model.add(Dropout(0.5))
	#model.add(LeakyReLU())
	model.add(Dense(depth*dim*dim,input_shape=noise_shape))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(Reshape((dim,dim,depth)))
	model.add(Conv2DTranspose(depth,(5,5),strides=2,padding='same'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization(axis=-1))
	model.add(LeakyReLU())
	model.add(Convolution2D(int(depth/2),(5,5),padding='same'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization(axis=-1))
	model.add(LeakyReLU())
	model.add(Conv2DTranspose(int(depth/2),(5,5),strides=2,padding='same'))
	model.add(Dropout(0.5))
	model.add(LeakyReLU())
	model.add(Convolution2D(1,(5,5),padding='same',activation='tanh'))

	model.summary()

        noise = Input(shape=noise_shape)
        label = Input(shape=(1,),dtype='int32')

	label_embedding = Flatten()(Embedding(self.num_classes,self.noise_shape)(label))
	model_input = multiply([noise, label_embedding])
	
	img = model(model_input)

        return Model([noise, label], img)



    def build_generator_dense(self):
	model = Sequential()
	model.add(Dense(128, input_dim=self.latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(256))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(512))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(np.prod(self.img_shape), activation='tanh'))
	model.add(Reshape(self.img_shape))

	model.summary()

	noise = Input(shape=(self.latent_dim,))
	label = Input(shape=(1,), dtype='int32')

	label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

	model_input = multiply([noise, label_embedding])

	img = model(model_input)

	return Model([noise, label], img)


    def build_discriminator_dense(self):
	model = Sequential()	
	model.add(Dense(128, input_dim=np.prod(self.img_shape)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dense(128))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Dense(128))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='tanh'))
	model.summary()

	img = Input(shape=self.img_shape)
	label = Input(shape=(1,), dtype='int32')

	label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
	flat_img = Flatten()(img)

	model_input = multiply([flat_img, label_embedding])

	validity = model(model_input)

	return Model([img, label], validity)

    def build_generator(self):

        noise_shape = (100,)
	dim = 50
        model = Sequential()
	depth = 64
        model.add(Dense(depth * dim * dim, activation="relu", input_shape=noise_shape))
        model.add(Reshape((dim, dim, depth)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(depth, kernel_size=4, padding="same"))
	model.add(Dropout(0.5))
	model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(int(depth/2), kernel_size=4, padding="same"))
	model.add(Dropout(0.5))
	model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Dropout(0.5))
	model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        label = Input(shape=(1,),dtype='int32')

	label_embedding = Flatten()(Embedding(self.num_classes,self.noise_shape)(label))
	model_input = multiply([noise, label_embedding])
	
	img = model(model_input)
	print(img.shape)
        return Model([noise, label], img)



    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()
	model.add(Reshape(img_shape,input_shape=(self.img_rows*self.img_cols*self.channels,) ))
        model.add(Conv2D(16, kernel_size=3, strides=2,padding="same")) 
        model.add(Dropout(0.4))
	model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1)))) 
        model.add(Dropout(0.4))
	model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(Dropout(0.4))
	model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same")) 
        model.add(Dropout(0.4))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dense(1,activation="linear"))
        model.add(Flatten())
        model.summary()
	
        img = Input(shape=img_shape)
        label = Input(shape=(1,),dtype='int32')
	
	label_embedding = Flatten()(Embedding(self.num_classes,np.prod(self.img_shape))(label))
	flat_img = Flatten()(img)

	model_input = multiply([flat_img, label_embedding])
	validity = model(model_input) 
        return Model([img, label], validity)


    def build_discriminator_exmpl(self):
	img_shape = (self.img_rows,self.img_cols,self.channels)
	depth = 64
	model = Sequential()
	model.add(Reshape(img_shape,input_shape=(self.img_rows*self.img_cols*self.channels,)))
	model.add(Convolution2D(int(depth/2),(5,5),padding='same'))
	model.add(Dropout(0.6))
	model.add(LeakyReLU())
	model.add(Convolution2D(depth,(5,5),kernel_initializer='he_normal',strides=[2,2]))
	model.add(Dropout(0.6))
	model.add(LeakyReLU())
	model.add(Convolution2D(depth,(5,5),kernel_initializer='he_normal',padding='same',strides=[2,2]))
	model.add(Dropout(0.6))
	model.add(LeakyReLU())
	model.add(Flatten())
	#model.add(Dense(depth,kernel_initializer='he_normal'))
	#model.add(LeakyReLU())
	model.add(Dense(1,kernel_initializer='he_normal'))
	
	img = Input(shape=img_shape)
        label = Input(shape=(1,),dtype='int32')
	
	label_embedding = Flatten()(Embedding(self.num_classes,np.prod(self.img_shape))(label))
	flat_img = Flatten()(img)

	model_input = multiply([flat_img, label_embedding])
	validity = model(model_input) 
        return Model([img, label], validity)



    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        (X_train, y_train) = read_utk_face('UTKFace/',size=(self.img_rows,self.img_cols))

	#(X_train,y_train),(_,_) = mnist.load_data()
	#(X_train,y_train) = read_utk_face('UTKFace')
	
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
	y_train = y_train.reshape(-1,1) #if using read_utk_face data
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
	writer = tf.summary.FileWriter('logdir')
        for epoch in range(epochs):
            idxs = []
	    noises = []
	    for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------
			


		
                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                idxs.append(idx)
		imgs = X_train[idx]
		labels_train = y_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, 100))
                noises.append(noise)
		# Train the discriminator
		
                d_loss = self.discriminator_model.train_on_batch([imgs, noise,labels_train],
                                                                [valid, fake, dummy])

	
            # ---------------------
            #  Train Generator
            # ---------------------
		
            # Sample generator input
            # Train the generator
            noise_gen = np.random.normal(0,1, (batch_size, 100))
	
	    #labels_train = y_train[idxs[4]]
	    labels_train = np.random.randint(0,10,batch_size)  
	    g_loss = self.generator_model.train_on_batch([noise_gen,labels_train],[valid])


            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss))
	    summary_g = tf.Summary(value=[tf.Summary.Value(tag='cwgan_G_loss',simple_value=(1-g_loss)),])
	    summary_d = tf.Summary(value=[tf.Summary.Value(tag='cwgan_D_loss',simple_value=(1-d_loss[0])),])
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
		self.discriminator_model.save_weights('discriminator_weights.h5')
		self.generator_model.save_weights('generator_weights.h5')
	    writer.add_summary(summary_d)
	    writer.add_summary(summary_g)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (10, 100))
	sampled_labels = np.arange(0,10).reshape(-1,1)
        gen_imgs = self.generator.predict([noise,sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        cnt = 0
        for i in range(0,10):
		    dt = np.squeeze(gen_imgs)
		    dt = dt[cnt,:,:]
		    rescaled = (255.0 / dt.max() * (dt - dt.min())).astype(np.uint8)
		    img = Image.fromarray(np.squeeze(rescaled))
		    img.save('Results_cwgan/' + str(epoch) + '_genimg_' + str(sampled_labels[cnt]) + '.png')
		    cnt += 1


if __name__ == '__main__':
    wgan = ImprovedWGAN()
    wgan.train(epochs=10000, batch_size=32, sample_interval=100)

import tensorflow as tf

class Dna2Vec(tf.keras.Model):
    def __init__(self, vocabulary, embedding_dimension=32):
        super(Dna2Vec, self).__init__()
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.embedding_dimension = embedding_dimension

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=embedding_dimension,
            mask_zero=True,  # equivalent to padding_idx in PyTorch
            embeddings_initializer=tf.keras.initializers.Constant(self.vocabulary["pad"]),
            name='embedding'
        )

        self.linear = tf.keras.layers.Dense(
            units=self.vocabulary_size,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            name='linear'
        )

    def call(self, context):
        embeds = self.embedding(context)
        mean_embeds = tf.reduce_mean(embeds, axis=1)
        output = self.linear(mean_embeds)
        return output
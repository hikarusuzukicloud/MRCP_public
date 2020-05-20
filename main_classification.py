import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tf.random.set_seed(42)

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [299, 299])
  image /= 255.0  # normalize to [0,1] range
  return image

class TFDataset():
    def __init__(self, ds, batch_size):
        self.batch_size = batch_size 
        self.iterator = None
        self.shuffle_buffer = 200
        self.dataset = ds.batch(batch_size)
        self.get_iterator()

    def get_iterator(self):
        self.iterator = self.dataset.__iter__()

    def reset_iterator(self):
        self.dataset.shuffle(self.shuffle_buffer)
        self.get_iterator()

    def get_batch(self):
        while True:
            try:
                batch = self.iterator.next()
                pts = batch[0]
                label = batch[1]
                break
            except:
                self.reset_iterator()
        return pts, label

class TransferNet(tf.keras.Model):
   def __init__(self, IMG_X_SIZE, IMG_Y_SIZE, N_CLASSES):
        super(TransferNet, self).__init__()
        self.IMG_X_SIZE = IMG_X_SIZE
        self.IMG_Y_SIZE = IMG_Y_SIZE
        self.N_CLASSES = N_CLASSES
        self.trainable = False

        self.init_network()

   def init_network(self):
        self.base_model = tf.keras.applications.Xception(input_shape=(self.IMG_Y_SIZE, self.IMG_X_SIZE, 3),
        #self.base_model = tf.keras.applications.VGG16(input_shape=(self.IMG_Y_SIZE, self.IMG_X_SIZE, 3),
                                                include_top=False,
                                                weights='imagenet')
        self.base_model.trainable = self.trainable
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.fc_layer = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.prediction_layer = tf.keras.layers.Dense(self.N_CLASSES, activation=tf.nn.softmax)

   def call(self, x):       
        feature_batch = self.base_model(x) 
        feature_batch_average = self.global_average_layer(feature_batch)
        feature_batch_fc = self.fc_layer(feature_batch_average)
        prediction_batch = self.prediction_layer(feature_batch_fc)
        return prediction_batch

def train_step(optimizer, model, loss_object, train_loss, train_acc, 
                image_batch, train_labels):
    with tf.GradientTape() as tape:
        pred = model(image_batch)
        loss = loss_object(train_labels, pred)

    train_loss.update_state([loss])
    train_acc.update_state(train_labels, pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_loss, train_acc

def main(config, params):
    img_filenames = os.listdir(config["dataset_dir"])
    all_image_paths = [os.path.join(config["dataset_dir"], p) for p in img_filenames]
    labels = []
    for s in img_filenames:
        if "AIP" in s:
            labels.append(0)
        else:
            labels.append(1)
    image_num = len(all_image_paths)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=params["autotune"])
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    train_ds = TFDataset(image_label_ds, params["batch_size"])

    model = TransferNet(params["image_size"], params["image_size"], params["class_num"])
    #model = TransferNet(299, 299, params["class_num"])
    model.build(input_shape=(params["batch_size"], params["image_size"], params["image_size"], 3))

    optimizer = tf.keras.optimizers.Adam(params["lr"])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    train_summary_writer = tf.summary.create_file_writer(config["logdir"])
        #os.path.join(config['log_dir'], config['log_code'], 'train')
    epochs = 10
    template = 'Epoch {}\n Loss: {}, Accuracy: {}\n Test Loss: {}, Test Accuracy: {}'
    while True:
        image_batch, label_batch = train_ds.get_batch()
        train_loss, train_acc = train_step(
            optimizer,
            model,
            loss_object,
            train_loss,
            train_acc,
            image_batch,
            label_batch
        )
        print(optimizer.iterations)
        if optimizer.iterations % config["log_freq"] == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                tf.summary.scalar('accuracy', train_acc.result(), step=optimizer.iterations)
            print(template.format(int(optimizer.iterations),
                                  train_loss.result(), 
                                  train_acc.result()*100,
                                  test_loss.result(), 
                                  test_acc.result()*100))   
            if optimizer.iterations % config["checkpoint_freq"] == 0:
                checkpointpath = "checkpoint_xception_%08d" % (optimizer.iterations)
                checkpointpath = os.path.join(config["checkpointdir"], checkpointpath)
                model.save_weights(checkpointpath)
                if int(optimizer.iterations) == config["finish_steps"]:
                    break
if __name__=="__main__":
    train_name = "teat"
    config = {
            "dataset_dir" : "/media/hikaru/ubuntuhdd/mrcp_dataset/images_before_tfrecord/ALL_original_jpg",
            "logdir" : "/media/hikaru/ubuntuhdd/mrcp_dataset/classification/classification_log/" + train_name,
            "log_freq" : 10,
            "checkpointdir" : "",
            "checkpoint_freq" : 100,
            "finish_steps" : 1000,
    }
    #module_url = "https://tfhub.dev/tensorflow/efficientnet/b4/classification/1"
    #module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    #module_url = "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"
    params = {
            "image_size" : 299, #preprocess_image func. change too
            "batch_size" : 24,
            "class_num" : 2,
            "lr" : 1e-3,
            "autotune" : tf.data.experimental.AUTOTUNE,
            }
    main(config, params)

import tensorflow.compat.v1 as tf
import os

class NutritionTextDetector:
    def __init__(self):
        model_path = os.path.join('data', 'models', 'ctpn.pb')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

            self.input_img = self.sess.graph.get_tensor_by_name('Placeholder:0')
            self.output_cls_prob = self.sess.graph.get_tensor_by_name('Reshape_2:0')
            self.output_box_pred = self.sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

        self.sess = tf.Session(graph=self.detection_graph)

    def get_text_classification(self, blobs):
        """Bounding Box Detection."""
        with self.detection_graph.as_default():
            (cls_prob, box_pred) = self.sess.run(
                [self.output_cls_prob, self.output_box_pred],
                feed_dict={self.input_img: blobs['data']})

        return cls_prob, box_pred

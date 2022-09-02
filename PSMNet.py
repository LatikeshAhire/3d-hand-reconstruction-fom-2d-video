# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import config


class PSMNet:

    def __init__(self, width, height, channels, batch_size, head_type):
        self.img_width = width
        self.img_height = height
        self.channels = channels
        self.batch_size = batch_size
        self.head_type = head_type
        # Model input left and right two pictures
        self.left_inputs = tf.placeholder(tf.float32,
                                          (None, self.img_height, self.img_width, self.channels), name='left_inputs')
        self.right_inputs = tf.placeholder(tf.float32,
                                           (None, self.img_height, self.img_width, self.channels), name='right_inputs')
        self.groundtruth = tf.placeholder(tf.float32,
                                          (None, self.img_height, self.img_width), name='groundtruth_disparity')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.optimizer = tf.train.AdamOptimizer(config.TRAIN_LR)

    def build_net(self):
        # Feature extraction Note that the right and left branches share weights when building a map
        self.ref_feature = self.feature_extraction(self.left_inputs)
        self.target_feature = self.feature_extraction(self.right_inputs, weight_share=True)

        # calculate cost volume
        self.cost_volume = self.cost_volume_aggregation(self.ref_feature, self.target_feature, config.MAX_DISP)

        #Implement 3d CNN and disparity map estimation
        if self.head_type == config.HEAD_STACKED_HOURGLASS:
            self.disparity_1, self.disparity_2, self.disparity_3 = self.stacked_hourglass(self.cost_volume)
        elif self.head_type == config.HEAD_BASIC:
            self.disparity = self.basic(self.cost_volume)
        else:
            raise NotImplementedError('Head Type \'{}\' Not Supported!!!'.format(self.head_type))

        # Calculate loss
        self.loss = self.calc_loss(self.disparity_1, self.disparity_2, self.disparity_3, self.groundtruth)
        tf.summary.scalar("loss", self.loss)

        # optimization
        self.train_op = self.optimizer.minimize(self.loss)

        # log
        self.train_log = tf.summary.merge_all()

    def cnn(self, inputs, weight_share=False):
        with tf.variable_scope('CNN_BASE'):
            outputs = inputs
            with tf.variable_scope('conv0'):
                # The first layer of convolution + downsampling The second and third layers of convolution
                for layer_id in range(3):
                    outputs = self._build_conv_block(
                        outputs, tf.layers.conv2d, filters=32, kernel_size=3,
                        strides=2 if layer_id == 0 else 1,
                        reuse=weight_share, layer_name='conv0_{}'.format(layer_id + 1)
                    )

            # Two-layer residual connection

            with tf.variable_scope('conv1'):
                # Three-layer res convolution
                for layer_id in range(3):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=32, kernel_size=3,
                        reuse=weight_share, layer_name='res_conv1_{}'.format(layer_id + 1)
                    )

            with tf.variable_scope('conv2'):
                # The first layer is downsampled twice and includes projection, and the remaining 15 layers of normal residual connections
                for layer_id in range(16):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=64, kernel_size=3,
                        strides=2 if layer_id == 0 else 1, projection=(layer_id == 0),
                        reuse=weight_share, layer_name='res_conv2_{}'.format(layer_id + 1)
                    )

            # Two-layer atrous convolution
            with tf.variable_scope('conv3'):
                # The first layer contains the projection and the remaining 2 layers are normal residual connections
                for layer_id in range(3):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=128, kernel_size=3,
                        dilation_rate=2, projection=(layer_id == 0),
                        reuse=weight_share, layer_name='res_conv3_{}'.format(layer_id + 1)
                    )

            with tf.variable_scope('conv4'):
                #Three-layer res-block
                for layer_id in range(3):
                    outputs = self._build_residual_block(
                        outputs, tf.layers.conv2d, filters=128, kernel_size=3,
                        dilation_rate=4,
                        reuse=weight_share, layer_name='res_conv4_{}'.format(layer_id + 1)
                    )

            return outputs

    def spp(self, inputs, weight_share):
        """
        Spatial Pyramid Module
        :param inputs:enter
        :param weight_share: weight sharing
        :return: context features

        """
        with tf.variable_scope('SPP'):
            # four branches of spp
            branches = [self._build_spp_branch(inputs, pool_size=pool_size,
                                               reuse=weight_share, layer_name='branch_{}'.format(branch_id + 1))
                        for branch_id, pool_size in enumerate([64, 32, 16, 8])]

            # Plus the skip connection conv2_16 and conv4_3 (ie inputs) in the CNN base Note that the connection before relu is added here
            branches.append(tf.get_default_graph().get_tensor_by_name('CNN_BASE/conv2/res_conv2_16/add:0'))
            branches.append(tf.get_default_graph().get_tensor_by_name('CNN_BASE/conv4/res_conv4_3/add:0'))

            # splicing
            outputs = tf.concat(branches, axis=-1, name='spp_branch_concat')

            # 　Feature fusion
            fusion = self._build_conv_block(outputs, tf.layers.conv2d, filters=128, kernel_size=3,
                                            reuse=weight_share, layer_name='fusion_conv_3x3')
            fusion = self._build_conv_block(fusion, tf.layers.conv2d, filters=32, kernel_size=1,
                                            reuse=weight_share, layer_name='fusion_conv_1x1')
            return fusion

    def feature_extraction(self, inputs, weight_share=False):
        """
        Feature extraction layer
        :param inputs: enter
        :param weight_share: Whether to share weights
        :return: feature
        """
        return self.spp(
            self.cnn(inputs, weight_share),
            weight_share
        )

    def cost_volume_aggregation(self, left_inputs, right_inputs, max_disp):
        """
        Aggregate the features of the left and right frames to generate cost-volume
        :param left_inputs: left input
        :param right_inputs: Right input
        :param max_disp: Maximum parallax depth
        :return: cost-volume
        """
        with tf.variable_scope('COST_VOLUME'):
            cost_volume = []
            for d in range(max_disp // 4):
                if d > 0:
                    # Parallax cost on the left and right sides Note here is the parallax calculated in the width dimension
                    left_shift = left_inputs[:, :, d:, :]
                    right_shift = left_inputs[:, :, :-d, :]

                    # Fill with 0 fill at the beginning of the dimension of width with 0
                    left_shift = tf.pad(left_shift, paddings=[[0, 0], [0, 0], [d, 0], [0, 0]])
                    right_shift = tf.pad(right_shift, paddings=[[0, 0], [0, 0], [d, 0], [0, 0]])

                    # Splicing in the channel dimension
                    cost_plate = tf.concat([left_shift, right_shift], axis=-1)
                else:
                    # When d is 0, directly splicing the original image without shift l dimension splicing
                    cost_plate = tf.concat([left_inputs, right_inputs], axis=-1)
                cost_volume.append(cost_plate)

            # Splicing the cost map of each parallax level into a cost volume Note that it should be spliced in the first dimension (the 0th is batch)
            cost_volume = tf.stack(cost_volume, axis=1)

        return cost_volume

    def stacked_hourglass(self, inputs):
        """
        stack hourglass for 3D convolution
        :param inputs: enter
        :return: Disparity prediction of 3 branches
        """
        with tf.variable_scope('ST_HGLS'):
            outputs = inputs
            # Two layers of ordinary 3D convolution
            with tf.variable_scope('3Dconv0'):
                for layer_id in range(2):
                    outputs = self._build_conv_block(outputs, tf.layers.conv3d, filters=32, kernel_size=3,
                                                     layer_name='3Dconv0_{}'.format(layer_id))
            with tf.variable_scope('3Dconv1'):
                _3Dconv1 = outputs = self._build_residual_block(outputs, tf.layers.conv3d, filters=32, kernel_size=3,
                                                                layer_name='res_3Dconv1')
            # three-layer stacked hourglass
            with tf.variable_scope('3Dstack1'):
                outputs, _3Dstack1_1, _3Dstack1_3 = self.hourglass(outputs, None, None, _3Dconv1, name='3Dstack1')
                # regression output
                disparity_1, classify_skip_out = self.disparity_regression(outputs, pre=None,
                                                                           name='output_1')

            with tf.variable_scope('3Dstack2'):
                outputs, _, _3Dstack2_3 = self.hourglass(outputs, _3Dstack1_3, _3Dstack1_1, _3Dconv1, name='3Dstack2')
                # Regression output plus one layer of skip
                disparity_2, classify_skip_out = self.disparity_regression(outputs, pre=classify_skip_out,
                                                                           name='output_2')

            with tf.variable_scope('3Dstack3'):
                outputs, _, _ = self.hourglass(outputs, _3Dstack2_3, _3Dstack1_1, _3Dconv1, name='3Dstack3')
                # Regression output plus one layer of skip
                disparity_3, _ = self.disparity_regression(outputs, pre=classify_skip_out,
                                                           name='output_3')

        return disparity_1, disparity_2, disparity_3

    def basic(self, inputs):
        return inputs

    def disparity_regression(self, inputs, pre, name):
        """
        Disparity map regression
        :param inputs:Input 3d cost volume feature
        :param pre: The output of the previous regression layer
        :param name: name
        :return: The disparity map obtained by regression, the skip output of the intermediate layer
        """
        with tf.variable_scope(name):
            with tf.variable_scope('classify'):
                # Ordinary 3d convolution
                outputs = self._build_conv_block(inputs, tf.layers.conv3d, filters=32, kernel_size=3,
                                                 layer_name='conv')
                # Aggregated to 1 channel and there is an intermediate skip connection going out
                classify_skip_out = outputs = \
                    self._build_conv_block(outputs, tf.layers.conv3d, filters=1, kernel_size=3,
                                           apply_bn=False, apply_relu=False, layer_name='conv_agg')
                # Add the output of the previous layer
                if pre is not None:
                    outputs = tf.add(outputs, pre, name='add')

            with tf.variable_scope('up_reg'):
                # Upsampling and regression
                # Squeeze the 1 channel of the last dimension
                outputs = tf.squeeze(outputs, [4])

                # Upsampling 4 times Note that this is the cost volume not the image, you need to use 3D upsampling
                # outputs = tf.keras.layers.UpSampling3D(size=4)(outputs)
                # Upsampling 4 times Fixed 3D upsampling bug
                outputs = tf.transpose(
                    tf.image.resize_images(
                        tf.transpose(outputs, perm=[0, 2, 3, 1])
                        , size=(self.img_height, self.img_width)
                    ), perm=[0, 3, 1, 2])

                # Regress cost to disparity map using soft-attention
                with tf.variable_scope('soft_attention'):
                    #Calculate the softmax of the original disparity map
                    logits_volume = tf.nn.softmax(outputs, axis=1)

                    # The weight of the dot product with logits_map is the increasing sequence of parallax
                    d_weight = tf.range(0, config.MAX_DISP, delta=4, dtype=tf.float32, name='d_weight')
                    # Here we need to expand the tile to the same dimension as logit_volume in order to perform broadcasting operations (multiply the parallax column corresponding to each pixel and d_weight)
                    d_weight = tf.tile(
                        tf.reshape(d_weight, shape=[1, config.MAX_DISP // 4, 1, 1]),
                        multiples=[tf.shape(logits_volume)[0], 1,
                                   logits_volume.shape[2].value, logits_volume.shape[3].value]
                    )

                    # product
                    disparity = tf.reduce_sum(
                        tf.multiply(logits_volume, d_weight),
                        axis=1,
                        name='soft_attention_dot'
                    )

            return disparity, classify_skip_out

    def hourglass(self, inputs, shortcut_1, shortcut_2, shortcut_3, name):
        """
        # Build the hourglass block
        :param inputs: previous layer input
        :param shortcut_1: 3Dstack(1,2)_3
        :param shortcut_2: 3Dstack1_1
        :param shortcut_3: 3Dconv1
        :param name: name
        :return: Output, skip output of stackX_1, skip output of stackX_3
        """
        with tf.variable_scope(name + '_1'):
            # first layer downsampling
            outputs = self._build_conv_block(inputs, tf.layers.conv3d, filters=64, kernel_size=3,
                                             strides=2, layer_name='downsample')
            outputs = self._build_conv_block(outputs, tf.layers.conv3d, filters=64, kernel_size=3,
                                             apply_relu=False, layer_name='3Dconv')
            if shortcut_1 is not None:
                # After the first layer of the stack, add the previous sortcut. Note that stack1_1 does not need to add a shortcut.
                outputs = tf.add(outputs, shortcut_1, name='add')

            # After the skip connection is added, relu and output as a shortcut
            skip_out_1 = outputs = tf.nn.relu(outputs, name='relu')

        with tf.variable_scope(name + '_2'):
            # Second layer downsampling
            outputs = self._build_conv_block(outputs, tf.layers.conv3d, filters=64, kernel_size=3,
                                             strides=2, layer_name='downsample')
            outputs = self._build_conv_block(outputs, tf.layers.conv3d, filters=64, kernel_size=3,
                                             layer_name='3Dconv')

        with tf.variable_scope(name + '_3'):
            # Upsampling transposed convolution
            outputs = self._build_conv_block(outputs, tf.layers.conv3d_transpose, filters=64, kernel_size=3,
                                             strides=2, apply_relu=False, layer_name='3Ddeconv')
            # add skip connection
            if shortcut_2 is not None:
                # If it is the parameter shortcut_2 passed in from other hourglass
                outputs = tf.add(outputs, shortcut_2, name='add')
            else:
                # If it is in hourglass1, directly add _1 of this layer
                outputs = tf.add(outputs, skip_out_1, name='add')

            # After the skip connection is added, relu and output as a shortcut
            skip_out_2 = outputs = tf.nn.relu(outputs, name='relu')

        with tf.variable_scope(name + '_4'):
            # Upsampling transposed convolution
            outputs = self._build_conv_block(outputs, tf.layers.conv3d_transpose, filters=32, kernel_size=3,
                                             strides=2, apply_relu=False, layer_name='3Ddeconv')
            # result without relu
            outputs = tf.add(outputs, shortcut_3, name='add')

        return outputs, skip_out_1, skip_out_2

    def calc_loss(self, disparity_1, disparity_2, disparity_3, groundtruth):
        """
        Calculate the total loss
        :param disparity_1: branch 1 disparity map
        :param disparity_2: branch 2 disparity map
        :param disparity_3: branch 3 disparity map
        :param groundtruth: label
        :return: Branch total loss3 disparity map
        """
        with tf.variable_scope('LOSS'):
            loss_coef = config.TRAIN_LOSS_COEF
            loss = loss_coef[0] * self._smooth_l1_loss(disparity_1, groundtruth) \
                   + loss_coef[1] * self._smooth_l1_loss(disparity_2, groundtruth) \
                   + loss_coef[2] * self._smooth_l1_loss(disparity_3, groundtruth)
        return loss

    def _build_conv_block(self, inputs, conv_function, filters, kernel_size, strides=1, dilation_rate=None,
                          layer_name='conv', apply_bn=True, apply_relu=True, reuse=False):
        # Building Convolutional Blocks
        conv_param = {
            'padding': 'same',
            'kernel_initializer': tf.keras.initializers.glorot_normal(),
            'kernel_regularizer': tf.keras.regularizers.L2(config.L2_REG),
            'bias_regularizer': tf.keras.regularizers.L2(config.L2_REG),
            'reuse': reuse
        }
        if dilation_rate:
            conv_param['dilation_rate'] = dilation_rate
        # Building Convolutional Blocks
        with tf.variable_scope(layer_name):
            # convolution
            outputs = conv_function(inputs, filters, kernel_size, strides, **conv_param)

            # bn
            if apply_bn:
                outputs = tf.layers.batch_normalization(
                    outputs, training=tf.get_default_graph().get_tensor_by_name('is_training:0'),
                    reuse=reuse, name='bn'
                )

            # activation function
            if apply_relu:
                outputs = tf.nn.relu(outputs)

            return outputs

    def _build_residual_block(self, inputs, conv_function, filters, kernel_size, strides=1, dilation_rate=None,
                              layer_name='conv', reuse=False, projection=False):
        # Build Residual Connection Blocks
        with tf.variable_scope(layer_name):
            inputs_shortcut = inputs
            # Build the first two conv layers of res_block
            outputs = self._build_conv_block(inputs, conv_function, filters, kernel_size, strides=strides,
                                             dilation_rate=dilation_rate, layer_name=layer_name + '_1', reuse=reuse)

            # Note that the second layer has no relu and strides=1 (guarantee no downsampling, downsampling is done by the first conv)
            outputs = self._build_conv_block(outputs, conv_function, filters, kernel_size, strides=1,
                                             dilation_rate=dilation_rate, layer_name=layer_name + '_2',
                                             apply_relu=False, reuse=reuse)

            # 1x1 projection to ensure that the channels of inputs_shortcut and outputs are consistent
            if projection:
                inputs_shortcut = self._build_conv_block(inputs_shortcut, conv_function, filters, kernel_size=1,
                                                         strides=strides, layer_name='projection',
                                                         apply_relu=False, apply_bn=False, reuse=reuse)
            # Add residual connection

            outputs = tf.add(outputs, inputs_shortcut, name='add')
            outputs = tf.nn.relu(outputs, name='relu')
            return outputs

    def _build_spp_branch(self, inputs, pool_size, reuse, layer_name):
        # Build a branch of spp
        with tf.variable_scope(layer_name):
            # original size
            origin_size = tf.shape(inputs)[1:3]

            # average pooling

            outputs = tf.layers.average_pooling2d(inputs, pool_size, strides=1, name='avg_pool')

            # convolution
            outputs = self._build_conv_block(outputs, tf.layers.conv2d, filters=32, kernel_size=3, reuse=reuse)

            # Upsampling Restore original size
            outputs = tf.image.resize_images(outputs, size=origin_size)

            return outputs

    def _smooth_l1_loss(self, estimation, groundtruth):
        # Calculate smooth l1 loss
        # https://github.com/rbgirshick/py-faster-rcnn/files/764206/SmoothL1Loss.1.pdf
        with tf.variable_scope('smooth_l1_loss'):
            # Calculate pixel difference
            diff = groundtruth - estimation
            abs_diff = tf.abs(diff)

            # According to the definition of sml1-loss, find the error less than the threshold. Note that the gradient is not forwarded here (equivalent to making a judgment)
            sign_mask = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1)), name='sign_mask')

            # Calculate the loss for each pixel

            smooth_l1_loss_map = \
                0.5 * tf.pow(diff, 2) * sign_mask \
                + (abs_diff - 0.5) * (1.0 - sign_mask)

            # Find the average loss of each pixel in all batches
            loss = tf.reduce_mean(tf.reduce_mean(smooth_l1_loss_map, axis=[1, 2]))
            # print(diff, abs_diff, sign_mask, smooth_l1_loss_map, loss)
        return loss

    def train(self, session: tf.Session, left_imgs, right_imgs, disp_gt):
        """
        train
        :param session: tf.session
        :param left_imgs:left view batch
        :param right_imgs: Right view batch
        :param disp_gt: Parallax groundtruth
        :return: loss
        """
        # optimize and forward
        loss, _, log = session.run(
            [self.loss, self.train_op, self.train_log],
            feed_dict={
                self.left_inputs: left_imgs,
                self.right_inputs: right_imgs,
                self.groundtruth: disp_gt,
                self.is_training: True
            }
        )
        return loss, log

    def predict(self, session: tf.Session, left_imgs, right_imgs):
        """
        test
        :param session: tf.session
        :param left_imgs: left view batch
        :param right_imgs: Right view batch
        :return: prediction
        """
        prediction = session.run(
            self.disparity_3,
            feed_dict={
                self.left_inputs: left_imgs,
                self.right_inputs: right_imgs,
                self.is_training: True  # The problem here still needs to be set to true bn needs to be confirmed
            }
        )
        return prediction


if __name__ == '__main__':
    print(config.TRAIN_CROP_WIDTH, config.TRAIN_CROP_HEIGHT, )
    psm_net = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT,
                     head_type=config.HEAD_STACKED_HOURGLASS, channels=config.IMG_N_CHANNEL, batch_size=18)
    psm_net.build_net()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./log", sess.graph)

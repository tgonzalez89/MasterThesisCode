
Q
conv2d_54_inputPlaceholder*
dtype0*$
shape:?????????

?
conv2d_54/kernelConst*
dtype0*a
valueXBV"@???>??!?ά???܃??|@??,>R??@????UP6=?1¾?I????y???@}y@??@??i@
[
conv2d_54/biasConst*
dtype0*5
value,B*" ?Y???a???{???X]??$??r`
?????ܜ+?
?
conv2d_54/Conv2DConv2Dconv2d_54_inputconv2d_54/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
^
conv2d_54/BiasAddBiasAddconv2d_54/Conv2Dconv2d_54/bias*
T0*
data_formatNHWC
2
conv2d_54/ReluReluconv2d_54/BiasAdd*
T0
?
conv2d_55/kernelConst*
dtype0*a
valueXBV"@c???	<?.????B?=6֍?<?Ծn?Ͼ
??>C?(;?>.?U??;?>?j?M˞>?=?
C
conv2d_55/biasConst*
dtype0*
valueB"??Ƽ7@
?
conv2d_55/Conv2DConv2Dconv2d_54/Reluconv2d_55/kernel*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
^
conv2d_55/BiasAddBiasAddconv2d_55/Conv2Dconv2d_55/bias*
T0*
data_formatNHWC
2
conv2d_55/ReluReluconv2d_55/BiasAdd*
T0
¥¥
Ù
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68
§
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!separable_conv2d/depthwise_kernel
 
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*'
_output_shapes
:*
dtype0
§
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!separable_conv2d/pointwise_kernel
 
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*'
_output_shapes
:
*
dtype0

separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameseparable_conv2d/bias
{
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes
:
*
dtype0
ª
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#separable_conv2d_1/depthwise_kernel
£
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
:
*
dtype0
ª
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*4
shared_name%#separable_conv2d_1/pointwise_kernel
£
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*&
_output_shapes
:

*
dtype0

separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameseparable_conv2d_1/bias

+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv3/kernel

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
µ
(Adam/separable_conv2d/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/m
®
<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/m*'
_output_shapes
:*
dtype0
µ
(Adam/separable_conv2d/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/m
®
<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/m*'
_output_shapes
:
*
dtype0

Adam/separable_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/separable_conv2d/bias/m

0Adam/separable_conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/m*
_output_shapes
:
*
dtype0
¸
*Adam/separable_conv2d_1/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/m
±
>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/m*&
_output_shapes
:
*
dtype0
¸
*Adam/separable_conv2d_1/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/m
±
>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/m*&
_output_shapes
:

*
dtype0

Adam/separable_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/separable_conv2d_1/bias/m

2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/m*
_output_shapes
:
*
dtype0
µ
(Adam/separable_conv2d/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/v
®
<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/v*'
_output_shapes
:*
dtype0
µ
(Adam/separable_conv2d/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/v
®
<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/v*'
_output_shapes
:
*
dtype0

Adam/separable_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/separable_conv2d/bias/v

0Adam/separable_conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/v*
_output_shapes
:
*
dtype0
¸
*Adam/separable_conv2d_1/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/v
±
>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/v*&
_output_shapes
:
*
dtype0
¸
*Adam/separable_conv2d_1/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/v
±
>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/v*&
_output_shapes
:

*
dtype0

Adam/separable_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/separable_conv2d_1/bias/v

2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/v*
_output_shapes
:
*
dtype0
Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"ÅàÏÂÙéÂ)\÷Â

NoOpNoOp
Û
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bü
õ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	keras_api* 

	keras_api* 
å
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
 layer_with_weights-9
 layer-13
!layer-14
"layer_with_weights-10
"layer-15
#layer_with_weights-11
#layer-16
$layer_with_weights-12
$layer-17
%layer-18
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
¥
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses* 
Æ
3depthwise_kernel
4pointwise_kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
Æ
<depthwise_kernel
=pointwise_kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
¼
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_rate3mÉ4mÊ5mË<mÌ=mÍ>mÎ3vÏ4vÐ5vÑ<vÒ=vÓ>vÔ*
ú
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25
326
427
528
<29
=30
>31*
.
30
41
52
<3
=4
>5*
* 
°
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

iserving_default* 
* 
* 
* 
¦

Jkernel
Kbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*
¦

Lkernel
Mbias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses*

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
¨

Nkernel
Obias
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

Pkernel
Qbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

Rkernel
Sbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

Tkernel
Ubias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

Vkernel
Wbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses* 
¬

Xkernel
Ybias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses*
¬

Zkernel
[bias
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses*
¬

\kernel
]bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses*

¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses* 
¬

^kernel
_bias
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses*
¬

`kernel
abias
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses*
¬

bkernel
cbias
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses*

Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses* 
Ê
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25*
* 
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 
* 
{u
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41
52*

30
41
52*
* 

ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
}w
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1
>2*

<0
=1
>2*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
Ê
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25*
5
0
1
2
3
4
5
6*

ê0*
* 
* 
* 

J0
K1*
* 
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 

L0
M1*
* 
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 
* 
* 

N0
O1*
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

P0
Q1*
* 
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

R0
S1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

T0
U1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

V0
W1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses* 
* 
* 

X0
Y1*
* 
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 

Z0
[1*
* 
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses*
* 
* 

\0
]1*
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses* 
* 
* 

^0
_1*
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses*
* 
* 

`0
a1*
* 
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses*
* 
* 

b0
c1*
* 
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses* 
* 
* 
Ê
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25*

0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

Åtotal

Æcount
Ç	variables
È	keras_api*

J0
K1*
* 
* 
* 
* 

L0
M1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

N0
O1*
* 
* 
* 
* 

P0
Q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

R0
S1*
* 
* 
* 
* 

T0
U1*
* 
* 
* 
* 

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

X0
Y1*
* 
* 
* 
* 

Z0
[1*
* 
* 
* 
* 

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*
* 
* 
* 
* 

`0
a1*
* 
* 
* 
* 

b0
c1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Å0
Æ1*

Ç	variables*

VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/m\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/m\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/separable_conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/separable_conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/v\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/v\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/separable_conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/separable_conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_2Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿàà
È
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2Constblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_7561
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¹
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOp0Adam/separable_conv2d/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOp0Adam/separable_conv2d/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOpConst_1*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_8443
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biastotalcount(Adam/separable_conv2d/depthwise_kernel/m(Adam/separable_conv2d/pointwise_kernel/mAdam/separable_conv2d/bias/m*Adam/separable_conv2d_1/depthwise_kernel/m*Adam/separable_conv2d_1/pointwise_kernel/mAdam/separable_conv2d_1/bias/m(Adam/separable_conv2d/depthwise_kernel/v(Adam/separable_conv2d/pointwise_kernel/vAdam/separable_conv2d/bias/v*Adam/separable_conv2d_1/depthwise_kernel/v*Adam/separable_conv2d_1/pointwise_kernel/vAdam/separable_conv2d_1/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_8606Ðê

a
E__inference_block4_pool_layer_call_and_return_conditional_losses_5559

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÞU
ï
?__inference_vgg16_layer_call_and_return_conditional_losses_5808

inputs+
block1_conv1_5593:@
block1_conv1_5595:@+
block1_conv2_5610:@@
block1_conv2_5612:@,
block2_conv1_5628:@ 
block2_conv1_5630:	-
block2_conv2_5645: 
block2_conv2_5647:	-
block3_conv1_5663: 
block3_conv1_5665:	-
block3_conv2_5680: 
block3_conv2_5682:	-
block3_conv3_5697: 
block3_conv3_5699:	-
block4_conv1_5715: 
block4_conv1_5717:	-
block4_conv2_5732: 
block4_conv2_5734:	-
block4_conv3_5749: 
block4_conv3_5751:	-
block5_conv1_5767: 
block5_conv1_5769:	-
block5_conv2_5784: 
block5_conv2_5786:	-
block5_conv3_5801: 
block5_conv3_5803:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_5593block1_conv1_5595*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5592«
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_5610block1_conv2_5612*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5609ë
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_5523¡
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_5628block2_conv1_5630*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5627ª
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_5645block2_conv2_5647*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5644ì
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_5535¡
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_5663block3_conv1_5665*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_5662ª
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_5680block3_conv2_5682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_5679ª
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_5697block3_conv3_5699*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_5696ì
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_5547¡
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_5715block4_conv1_5717*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_5714ª
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_5732block4_conv2_5734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_5731ª
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_5749block4_conv3_5751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_5748ì
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_5559¡
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_5767block5_conv1_5769*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_5766ª
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_5784block5_conv2_5786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_5783ª
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_5801block5_conv3_5803*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_5800ì
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_5571|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs


F__inference_block3_conv1_layer_call_and_return_conditional_losses_5662

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs


F__inference_block4_conv3_layer_call_and_return_conditional_losses_8186

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
£
+__inference_block4_conv1_layer_call_fn_8135

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_5714x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
ô
?__inference_vgg16_layer_call_and_return_conditional_losses_7775

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88­
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs


F__inference_block5_conv2_layer_call_and_return_conditional_losses_8236

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

J__inference_separable_conv2d_layer_call_and_return_conditional_losses_7929

inputsC
(separable_conv2d_readvariableop_resource:E
*separable_conv2d_readvariableop_1_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:
*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
ß
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block1_pool_layer_call_and_return_conditional_losses_5523

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block5_conv3_layer_call_and_return_conditional_losses_5800

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_6435

inputsB
(separable_conv2d_readvariableop_resource:
D
*separable_conv2d_readvariableop_1_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:

*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
ß
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


F__inference_block5_conv1_layer_call_and_return_conditional_losses_5766

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
F
*__inference_block1_pool_layer_call_fn_8001

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_5523
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
_
A__inference_dropout_layer_call_and_return_conditional_losses_7890

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block3_conv2_layer_call_and_return_conditional_losses_8096

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
õ
£
+__inference_block3_conv2_layer_call_fn_8085

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_5679x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
õ
£
+__inference_block5_conv2_layer_call_fn_8225

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_5783x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
h

__inference__traced_save_8443
file_prefix@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopG
Csavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopG
Csavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop;
7savev2_adam_separable_conv2d_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_1_bias_m_read_readvariableopG
Csavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopG
Csavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop;
7savev2_adam_separable_conv2d_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_1_bias_v_read_readvariableop
savev2_const_1

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: §
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*Ð
valueÆBÃ4B@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B µ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop7savev2_adam_separable_conv2d_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_1_bias_m_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop7savev2_adam_separable_conv2d_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_1_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesï
ì: ::
:
:
:

:
: : : : : :@:@:@@:@:@:::::::::::::::::::::: : ::
:
:
:

:
::
:
:
:

:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::-)
'
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::. *
(
_output_shapes
::!!

_output_shapes	
::."*
(
_output_shapes
::!#

_output_shapes	
::.$*
(
_output_shapes
::!%

_output_shapes	
::&

_output_shapes
: :'

_output_shapes
: :-()
'
_output_shapes
::-))
'
_output_shapes
:
: *

_output_shapes
:
:,+(
&
_output_shapes
:
:,,(
&
_output_shapes
:

: -

_output_shapes
:
:-.)
'
_output_shapes
::-/)
'
_output_shapes
:
: 0

_output_shapes
:
:,1(
&
_output_shapes
:
:,2(
&
_output_shapes
:

: 3

_output_shapes
:
:4

_output_shapes
: 

¡	
6__inference_vgg16_keypoint_detector_layer_call_fn_6915
input_2
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@
	unknown_5:	%
	unknown_6:
	unknown_7:	%
	unknown_8:
	unknown_9:	&

unknown_10:

unknown_11:	&

unknown_12:

unknown_13:	&

unknown_14:

unknown_15:	&

unknown_16:

unknown_17:	&

unknown_18:

unknown_19:	&

unknown_20:

unknown_21:	&

unknown_22:

unknown_23:	&

unknown_24:

unknown_25:	%

unknown_26:%

unknown_27:


unknown_28:
$

unknown_29:
$

unknown_30:



unknown_31:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6775w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_2: 

_output_shapes
:
÷
±
$__inference_vgg16_layer_call_fn_5863
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_5808x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
»
B
&__inference_dropout_layer_call_fn_7880

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6515i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
°
$__inference_vgg16_layer_call_fn_7618

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_5808x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ö
 
+__inference_block1_conv1_layer_call_fn_7965

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5592y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

a
E__inference_block5_pool_layer_call_and_return_conditional_losses_5571

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÞU
ï
?__inference_vgg16_layer_call_and_return_conditional_losses_6126

inputs+
block1_conv1_6055:@
block1_conv1_6057:@+
block1_conv2_6060:@@
block1_conv2_6062:@,
block2_conv1_6066:@ 
block2_conv1_6068:	-
block2_conv2_6071: 
block2_conv2_6073:	-
block3_conv1_6077: 
block3_conv1_6079:	-
block3_conv2_6082: 
block3_conv2_6084:	-
block3_conv3_6087: 
block3_conv3_6089:	-
block4_conv1_6093: 
block4_conv1_6095:	-
block4_conv2_6098: 
block4_conv2_6100:	-
block4_conv3_6103: 
block4_conv3_6105:	-
block5_conv1_6109: 
block5_conv1_6111:	-
block5_conv2_6114: 
block5_conv2_6116:	-
block5_conv3_6119: 
block5_conv3_6121:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_6055block1_conv1_6057*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5592«
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6060block1_conv2_6062*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5609ë
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_5523¡
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_6066block2_conv1_6068*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5627ª
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6071block2_conv2_6073*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5644ì
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_5535¡
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_6077block3_conv1_6079*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_5662ª
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6082block3_conv2_6084*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_5679ª
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_6087block3_conv3_6089*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_5696ì
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_5547¡
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_6093block4_conv1_6095*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_5714ª
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_6098block4_conv2_6100*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_5731ª
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_6103block4_conv3_6105*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_5748ì
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_5559¡
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_6109block5_conv1_6111*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_5766ª
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_6114block5_conv2_6116*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_5783ª
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_6119block5_conv3_6121*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_5800ì
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_5571|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

ÿ
F__inference_block1_conv1_layer_call_and_return_conditional_losses_7976

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ÐÌ
é
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7488

inputs
tf_nn_bias_add_biasadd_biasK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	M
1vgg16_block2_conv2_conv2d_readvariableop_resource:A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv1_conv2d_readvariableop_resource:A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	M
1vgg16_block3_conv2_conv2d_readvariableop_resource:A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv3_conv2d_readvariableop_resource:A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	M
1vgg16_block4_conv1_conv2d_readvariableop_resource:A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	M
1vgg16_block4_conv2_conv2d_readvariableop_resource:A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	M
1vgg16_block4_conv3_conv2d_readvariableop_resource:A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	M
1vgg16_block5_conv1_conv2d_readvariableop_resource:A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	M
1vgg16_block5_conv2_conv2d_readvariableop_resource:A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	M
1vgg16_block5_conv3_conv2d_readvariableop_resource:A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	T
9separable_conv2d_separable_conv2d_readvariableop_resource:V
;separable_conv2d_separable_conv2d_readvariableop_1_resource:
>
0separable_conv2d_biasadd_readvariableop_resource:
U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:
W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:

@
2separable_conv2d_1_biasadd_readvariableop_resource:

identity¢'separable_conv2d/BiasAdd/ReadVariableOp¢0separable_conv2d/separable_conv2d/ReadVariableOp¢2separable_conv2d/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_1/BiasAdd/ReadVariableOp¢2separable_conv2d_1/separable_conv2d/ReadVariableOp¢4separable_conv2d_1/separable_conv2d/ReadVariableOp_1¢)vgg16/block1_conv1/BiasAdd/ReadVariableOp¢(vgg16/block1_conv1/Conv2D/ReadVariableOp¢)vgg16/block1_conv2/BiasAdd/ReadVariableOp¢(vgg16/block1_conv2/Conv2D/ReadVariableOp¢)vgg16/block2_conv1/BiasAdd/ReadVariableOp¢(vgg16/block2_conv1/Conv2D/ReadVariableOp¢)vgg16/block2_conv2/BiasAdd/ReadVariableOp¢(vgg16/block2_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv1/BiasAdd/ReadVariableOp¢(vgg16/block3_conv1/Conv2D/ReadVariableOp¢)vgg16/block3_conv2/BiasAdd/ReadVariableOp¢(vgg16/block3_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv3/BiasAdd/ReadVariableOp¢(vgg16/block3_conv3/Conv2D/ReadVariableOp¢)vgg16/block4_conv1/BiasAdd/ReadVariableOp¢(vgg16/block4_conv1/Conv2D/ReadVariableOp¢)vgg16/block4_conv2/BiasAdd/ReadVariableOp¢(vgg16/block4_conv2/Conv2D/ReadVariableOp¢)vgg16/block4_conv3/BiasAdd/ReadVariableOp¢(vgg16/block4_conv3/Conv2D/ReadVariableOp¢)vgg16/block5_conv1/BiasAdd/ReadVariableOp¢(vgg16/block5_conv1/Conv2D/ReadVariableOp¢)vgg16/block5_conv2/BiasAdd/ReadVariableOp¢(vgg16/block5_conv2/Conv2D/ReadVariableOp¢)vgg16/block5_conv3/BiasAdd/ReadVariableOp¢(vgg16/block5_conv3/Conv2D/ReadVariableOp}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿç
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_mask«
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà¢
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ú
vgg16/block1_conv1/Conv2DConv2Dtf.nn.bias_add/BiasAdd:output:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¢
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¸
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
£
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¤
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¹
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
¤
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout/dropout/MulMul"vgg16/block5_pool/MaxPool:output:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dropout/dropout/ShapeShape"vgg16/block5_pool/MaxPool:output:0*
T0*
_output_shapes
:¥
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0·
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:
*
dtype0
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ý
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativedropout/dropout/Mul_1:z:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0º
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0º
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:

*
dtype0
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
      
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0À
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
IdentityIdentity%separable_conv2d_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Û
NoOpNoOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs: 

_output_shapes
:
õ(
ÿ

Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6532

inputs
tf_nn_bias_add_biasadd_bias$

vgg16_6457:@

vgg16_6459:@$

vgg16_6461:@@

vgg16_6463:@%

vgg16_6465:@

vgg16_6467:	&

vgg16_6469:

vgg16_6471:	&

vgg16_6473:

vgg16_6475:	&

vgg16_6477:

vgg16_6479:	&

vgg16_6481:

vgg16_6483:	&

vgg16_6485:

vgg16_6487:	&

vgg16_6489:

vgg16_6491:	&

vgg16_6493:

vgg16_6495:	&

vgg16_6497:

vgg16_6499:	&

vgg16_6501:

vgg16_6503:	&

vgg16_6505:

vgg16_6507:	0
separable_conv2d_6517:0
separable_conv2d_6519:
#
separable_conv2d_6521:
1
separable_conv2d_1_6524:
1
separable_conv2d_1_6526:

%
separable_conv2d_1_6528:

identity¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall¢vgg16/StatefulPartitionedCall}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿç
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_mask«
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÐ
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0
vgg16_6457
vgg16_6459
vgg16_6461
vgg16_6463
vgg16_6465
vgg16_6467
vgg16_6469
vgg16_6471
vgg16_6473
vgg16_6475
vgg16_6477
vgg16_6479
vgg16_6481
vgg16_6483
vgg16_6485
vgg16_6487
vgg16_6489
vgg16_6491
vgg16_6493
vgg16_6495
vgg16_6497
vgg16_6499
vgg16_6501
vgg16_6503
vgg16_6505
vgg16_6507*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_5808Ý
dropout/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6515Å
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0separable_conv2d_6517separable_conv2d_6519separable_conv2d_6521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_6406à
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_6524separable_conv2d_1_6526separable_conv2d_1_6528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_6435
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¾
NoOpNoOp)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs: 

_output_shapes
:
·

`
A__inference_dropout_layer_call_and_return_conditional_losses_7902

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
£
+__inference_block4_conv3_layer_call_fn_8175

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_5748x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block4_conv1_layer_call_and_return_conditional_losses_8146

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
	
"__inference_signature_wrapper_7561
input_2
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@
	unknown_5:	%
	unknown_6:
	unknown_7:	%
	unknown_8:
	unknown_9:	&

unknown_10:

unknown_11:	&

unknown_12:

unknown_13:	&

unknown_14:

unknown_15:	&

unknown_16:

unknown_17:	&

unknown_18:

unknown_19:	&

unknown_20:

unknown_21:	&

unknown_22:

unknown_23:	&

unknown_24:

unknown_25:	%

unknown_26:%

unknown_27:


unknown_28:
$

unknown_29:
$

unknown_30:



unknown_31:

identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_5514w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_2: 

_output_shapes
:


F__inference_block3_conv3_layer_call_and_return_conditional_losses_8116

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs


F__inference_block2_conv1_layer_call_and_return_conditional_losses_8026

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@
 
_user_specified_nameinputs
¬
F
*__inference_block2_pool_layer_call_fn_8051

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_5535
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
£
+__inference_block5_conv1_layer_call_fn_8205

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_5766x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
±
$__inference_vgg16_layer_call_fn_6238
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_6126x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
®
ô
?__inference_vgg16_layer_call_and_return_conditional_losses_7875

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88­
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
õ
£
+__inference_block5_conv3_layer_call_fn_8245

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_5800x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block5_pool_layer_call_and_return_conditional_losses_8266

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
¢
+__inference_block2_conv1_layer_call_fn_8015

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5627x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@
 
_user_specified_nameinputs


F__inference_block3_conv2_layer_call_and_return_conditional_losses_5679

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ø
_
A__inference_dropout_layer_call_and_return_conditional_losses_6515

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5609

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
 
_user_specified_nameinputs


F__inference_block2_conv2_layer_call_and_return_conditional_losses_8046

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
¬
F
*__inference_block4_pool_layer_call_fn_8191

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_5559
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
Ë
/__inference_separable_conv2d_layer_call_fn_7913

inputs"
unknown:$
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_6406
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block2_conv2_layer_call_and_return_conditional_losses_5644

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
¬
F
*__inference_block5_pool_layer_call_fn_8261

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_5571
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block2_pool_layer_call_and_return_conditional_losses_5535

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

 	
6__inference_vgg16_keypoint_detector_layer_call_fn_7219

inputs
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@
	unknown_5:	%
	unknown_6:
	unknown_7:	%
	unknown_8:
	unknown_9:	&

unknown_10:

unknown_11:	&

unknown_12:

unknown_13:	&

unknown_14:

unknown_15:	&

unknown_16:

unknown_17:	&

unknown_18:

unknown_19:	&

unknown_20:

unknown_21:	&

unknown_22:

unknown_23:	&

unknown_24:

unknown_25:	%

unknown_26:%

unknown_27:


unknown_28:
$

unknown_29:
$

unknown_30:



unknown_31:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6775w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs: 

_output_shapes
:


F__inference_block5_conv1_layer_call_and_return_conditional_losses_8216

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block1_pool_layer_call_and_return_conditional_losses_8006

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block2_pool_layer_call_and_return_conditional_losses_8056

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block4_pool_layer_call_and_return_conditional_losses_8196

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_7956

inputsB
(separable_conv2d_readvariableop_resource:
D
*separable_conv2d_readvariableop_1_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:

*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
ß
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ø(

Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6993
input_2
tf_nn_bias_add_biasadd_bias$

vgg16_6924:@

vgg16_6926:@$

vgg16_6928:@@

vgg16_6930:@%

vgg16_6932:@

vgg16_6934:	&

vgg16_6936:

vgg16_6938:	&

vgg16_6940:

vgg16_6942:	&

vgg16_6944:

vgg16_6946:	&

vgg16_6948:

vgg16_6950:	&

vgg16_6952:

vgg16_6954:	&

vgg16_6956:

vgg16_6958:	&

vgg16_6960:

vgg16_6962:	&

vgg16_6964:

vgg16_6966:	&

vgg16_6968:

vgg16_6970:	&

vgg16_6972:

vgg16_6974:	0
separable_conv2d_6978:0
separable_conv2d_6980:
#
separable_conv2d_6982:
1
separable_conv2d_1_6985:
1
separable_conv2d_1_6987:

%
separable_conv2d_1_6989:

identity¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall¢vgg16/StatefulPartitionedCall}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿè
&tf.__operators__.getitem/strided_sliceStridedSliceinput_25tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_mask«
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÐ
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0
vgg16_6924
vgg16_6926
vgg16_6928
vgg16_6930
vgg16_6932
vgg16_6934
vgg16_6936
vgg16_6938
vgg16_6940
vgg16_6942
vgg16_6944
vgg16_6946
vgg16_6948
vgg16_6950
vgg16_6952
vgg16_6954
vgg16_6956
vgg16_6958
vgg16_6960
vgg16_6962
vgg16_6964
vgg16_6966
vgg16_6968
vgg16_6970
vgg16_6972
vgg16_6974*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_5808Ý
dropout/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6515Å
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0separable_conv2d_6978separable_conv2d_6980separable_conv2d_6982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_6406à
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_6985separable_conv2d_1_6987separable_conv2d_1_6989*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_6435
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¾
NoOpNoOp)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_2: 

_output_shapes
:
÷
Ë
1__inference_separable_conv2d_1_layer_call_fn_7940

inputs!
unknown:
#
	unknown_0:


	unknown_1:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_6435
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


F__inference_block4_conv1_layer_call_and_return_conditional_losses_5714

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
°
$__inference_vgg16_layer_call_fn_7675

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_6126x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ö
 
+__inference_block1_conv2_layer_call_fn_7985

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5609y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
 
_user_specified_nameinputs

ÿ
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5592

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs


F__inference_block2_conv1_layer_call_and_return_conditional_losses_5627

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@
 
_user_specified_nameinputs
ýÄ
é
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7350

inputs
tf_nn_bias_add_biasadd_biasK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	M
1vgg16_block2_conv2_conv2d_readvariableop_resource:A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv1_conv2d_readvariableop_resource:A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	M
1vgg16_block3_conv2_conv2d_readvariableop_resource:A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv3_conv2d_readvariableop_resource:A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	M
1vgg16_block4_conv1_conv2d_readvariableop_resource:A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	M
1vgg16_block4_conv2_conv2d_readvariableop_resource:A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	M
1vgg16_block4_conv3_conv2d_readvariableop_resource:A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	M
1vgg16_block5_conv1_conv2d_readvariableop_resource:A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	M
1vgg16_block5_conv2_conv2d_readvariableop_resource:A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	M
1vgg16_block5_conv3_conv2d_readvariableop_resource:A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	T
9separable_conv2d_separable_conv2d_readvariableop_resource:V
;separable_conv2d_separable_conv2d_readvariableop_1_resource:
>
0separable_conv2d_biasadd_readvariableop_resource:
U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:
W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:

@
2separable_conv2d_1_biasadd_readvariableop_resource:

identity¢'separable_conv2d/BiasAdd/ReadVariableOp¢0separable_conv2d/separable_conv2d/ReadVariableOp¢2separable_conv2d/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_1/BiasAdd/ReadVariableOp¢2separable_conv2d_1/separable_conv2d/ReadVariableOp¢4separable_conv2d_1/separable_conv2d/ReadVariableOp_1¢)vgg16/block1_conv1/BiasAdd/ReadVariableOp¢(vgg16/block1_conv1/Conv2D/ReadVariableOp¢)vgg16/block1_conv2/BiasAdd/ReadVariableOp¢(vgg16/block1_conv2/Conv2D/ReadVariableOp¢)vgg16/block2_conv1/BiasAdd/ReadVariableOp¢(vgg16/block2_conv1/Conv2D/ReadVariableOp¢)vgg16/block2_conv2/BiasAdd/ReadVariableOp¢(vgg16/block2_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv1/BiasAdd/ReadVariableOp¢(vgg16/block3_conv1/Conv2D/ReadVariableOp¢)vgg16/block3_conv2/BiasAdd/ReadVariableOp¢(vgg16/block3_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv3/BiasAdd/ReadVariableOp¢(vgg16/block3_conv3/Conv2D/ReadVariableOp¢)vgg16/block4_conv1/BiasAdd/ReadVariableOp¢(vgg16/block4_conv1/Conv2D/ReadVariableOp¢)vgg16/block4_conv2/BiasAdd/ReadVariableOp¢(vgg16/block4_conv2/Conv2D/ReadVariableOp¢)vgg16/block4_conv3/BiasAdd/ReadVariableOp¢(vgg16/block4_conv3/Conv2D/ReadVariableOp¢)vgg16/block5_conv1/BiasAdd/ReadVariableOp¢(vgg16/block5_conv1/Conv2D/ReadVariableOp¢)vgg16/block5_conv2/BiasAdd/ReadVariableOp¢(vgg16/block5_conv2/Conv2D/ReadVariableOp¢)vgg16/block5_conv3/BiasAdd/ReadVariableOp¢(vgg16/block5_conv3/Conv2D/ReadVariableOp}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿç
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_mask«
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà¢
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ú
vgg16/block1_conv1/Conv2DConv2Dtf.nn.bias_add/BiasAdd:output:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¢
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¸
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
£
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¤
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¹
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
¤
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
dropout/IdentityIdentity"vgg16/block5_pool/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0·
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:
*
dtype0
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ý
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativedropout/Identity:output:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0º
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0º
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:

*
dtype0
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
      
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0À
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
IdentityIdentity%separable_conv2d_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Û
NoOpNoOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs: 

_output_shapes
:
*
¡
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6775

inputs
tf_nn_bias_add_biasadd_bias$

vgg16_6706:@

vgg16_6708:@$

vgg16_6710:@@

vgg16_6712:@%

vgg16_6714:@

vgg16_6716:	&

vgg16_6718:

vgg16_6720:	&

vgg16_6722:

vgg16_6724:	&

vgg16_6726:

vgg16_6728:	&

vgg16_6730:

vgg16_6732:	&

vgg16_6734:

vgg16_6736:	&

vgg16_6738:

vgg16_6740:	&

vgg16_6742:

vgg16_6744:	&

vgg16_6746:

vgg16_6748:	&

vgg16_6750:

vgg16_6752:	&

vgg16_6754:

vgg16_6756:	0
separable_conv2d_6760:0
separable_conv2d_6762:
#
separable_conv2d_6764:
1
separable_conv2d_1_6767:
1
separable_conv2d_1_6769:

%
separable_conv2d_1_6771:

identity¢dropout/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall¢vgg16/StatefulPartitionedCall}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿç
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_mask«
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÐ
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0
vgg16_6706
vgg16_6708
vgg16_6710
vgg16_6712
vgg16_6714
vgg16_6716
vgg16_6718
vgg16_6720
vgg16_6722
vgg16_6724
vgg16_6726
vgg16_6728
vgg16_6730
vgg16_6732
vgg16_6734
vgg16_6736
vgg16_6738
vgg16_6740
vgg16_6742
vgg16_6744
vgg16_6746
vgg16_6748
vgg16_6750
vgg16_6752
vgg16_6754
vgg16_6756*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_6126í
dropout/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6621Í
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0separable_conv2d_6760separable_conv2d_6762separable_conv2d_6764*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_6406à
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_6767separable_conv2d_1_6769separable_conv2d_1_6771*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_6435
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à
NoOpNoOp ^dropout/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs: 

_output_shapes
:

 	
6__inference_vgg16_keypoint_detector_layer_call_fn_7148

inputs
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@
	unknown_5:	%
	unknown_6:
	unknown_7:	%
	unknown_8:
	unknown_9:	&

unknown_10:

unknown_11:	&

unknown_12:

unknown_13:	&

unknown_14:

unknown_15:	&

unknown_16:

unknown_17:	&

unknown_18:

unknown_19:	&

unknown_20:

unknown_21:	&

unknown_22:

unknown_23:	&

unknown_24:

unknown_25:	%

unknown_26:%

unknown_27:


unknown_28:
$

unknown_29:
$

unknown_30:



unknown_31:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6532w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs: 

_output_shapes
:
áU
ð
?__inference_vgg16_layer_call_and_return_conditional_losses_6386
input_1+
block1_conv1_6315:@
block1_conv1_6317:@+
block1_conv2_6320:@@
block1_conv2_6322:@,
block2_conv1_6326:@ 
block2_conv1_6328:	-
block2_conv2_6331: 
block2_conv2_6333:	-
block3_conv1_6337: 
block3_conv1_6339:	-
block3_conv2_6342: 
block3_conv2_6344:	-
block3_conv3_6347: 
block3_conv3_6349:	-
block4_conv1_6353: 
block4_conv1_6355:	-
block4_conv2_6358: 
block4_conv2_6360:	-
block4_conv3_6363: 
block4_conv3_6365:	-
block5_conv1_6369: 
block5_conv1_6371:	-
block5_conv2_6374: 
block5_conv2_6376:	-
block5_conv3_6379: 
block5_conv3_6381:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_6315block1_conv1_6317*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5592«
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6320block1_conv2_6322*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5609ë
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_5523¡
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_6326block2_conv1_6328*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5627ª
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6331block2_conv2_6333*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5644ì
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_5535¡
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_6337block3_conv1_6339*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_5662ª
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6342block3_conv2_6344*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_5679ª
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_6347block3_conv3_6349*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_5696ì
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_5547¡
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_6353block4_conv1_6355*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_5714ª
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_6358block4_conv2_6360*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_5731ª
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_6363block4_conv3_6365*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_5748ì
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_5559¡
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_6369block5_conv1_6371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_5766ª
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_6374block5_conv2_6376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_5783ª
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_6379block5_conv3_6381*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_5800ì
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_5571|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
ð
Ð*
__inference__wrapped_model_5514
input_27
3vgg16_keypoint_detector_tf_nn_bias_add_biasadd_biasc
Ivgg16_keypoint_detector_vgg16_block1_conv1_conv2d_readvariableop_resource:@X
Jvgg16_keypoint_detector_vgg16_block1_conv1_biasadd_readvariableop_resource:@c
Ivgg16_keypoint_detector_vgg16_block1_conv2_conv2d_readvariableop_resource:@@X
Jvgg16_keypoint_detector_vgg16_block1_conv2_biasadd_readvariableop_resource:@d
Ivgg16_keypoint_detector_vgg16_block2_conv1_conv2d_readvariableop_resource:@Y
Jvgg16_keypoint_detector_vgg16_block2_conv1_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block2_conv2_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block2_conv2_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block3_conv1_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block3_conv1_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block3_conv2_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block3_conv2_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block3_conv3_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block3_conv3_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block4_conv1_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block4_conv1_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block4_conv2_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block4_conv2_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block4_conv3_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block4_conv3_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block5_conv1_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block5_conv1_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block5_conv2_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block5_conv2_biasadd_readvariableop_resource:	e
Ivgg16_keypoint_detector_vgg16_block5_conv3_conv2d_readvariableop_resource:Y
Jvgg16_keypoint_detector_vgg16_block5_conv3_biasadd_readvariableop_resource:	l
Qvgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_resource:n
Svgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_1_resource:
V
Hvgg16_keypoint_detector_separable_conv2d_biasadd_readvariableop_resource:
m
Svgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_resource:
o
Uvgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:

X
Jvgg16_keypoint_detector_separable_conv2d_1_biasadd_readvariableop_resource:

identity¢?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp¢Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp¢Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1¢Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp¢Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp¢Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1¢Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp¢Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp¢@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp
Dvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Fvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Fvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿÈ
>vgg16_keypoint_detector/tf.__operators__.getitem/strided_sliceStridedSliceinput_2Mvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack:output:0Ovgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_1:output:0Ovgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_maskó
.vgg16_keypoint_detector/tf.nn.bias_add/BiasAddBiasAddGvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice:output:03vgg16_keypoint_detector_tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÒ
@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0¢
1vgg16_keypoint_detector/vgg16/block1_conv1/Conv2DConv2D7vgg16_keypoint_detector/tf.nn.bias_add/BiasAdd:output:0Hvgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
È
Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
2vgg16_keypoint_detector/vgg16/block1_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@°
/vgg16_keypoint_detector/vgg16/block1_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Ò
@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¨
1vgg16_keypoint_detector/vgg16/block1_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block1_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
È
Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
2vgg16_keypoint_detector/vgg16/block1_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@°
/vgg16_keypoint_detector/vgg16/block1_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@è
1vgg16_keypoint_detector/vgg16/block1_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
Ó
@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¤
1vgg16_keypoint_detector/vgg16/block2_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block1_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block2_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¯
/vgg16_keypoint_detector/vgg16/block2_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÔ
@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block2_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block2_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block2_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¯
/vgg16_keypoint_detector/vgg16/block2_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppé
1vgg16_keypoint_detector/vgg16/block2_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
Ô
@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¤
1vgg16_keypoint_detector/vgg16/block3_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block2_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block3_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¯
/vgg16_keypoint_detector/vgg16/block3_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Ô
@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block3_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block3_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block3_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¯
/vgg16_keypoint_detector/vgg16/block3_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Ô
@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block3_conv3/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block3_conv2/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block3_conv3/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¯
/vgg16_keypoint_detector/vgg16/block3_conv3/ReluRelu;vgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88é
1vgg16_keypoint_detector/vgg16/block3_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Ô
@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¤
1vgg16_keypoint_detector/vgg16/block4_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block3_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block4_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
/vgg16_keypoint_detector/vgg16/block4_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block4_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block4_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block4_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
/vgg16_keypoint_detector/vgg16/block4_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block4_conv3/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block4_conv2/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block4_conv3/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
/vgg16_keypoint_detector/vgg16/block4_conv3/ReluRelu;vgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
1vgg16_keypoint_detector/vgg16/block4_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Ô
@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¤
1vgg16_keypoint_detector/vgg16/block5_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block4_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block5_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
/vgg16_keypoint_detector/vgg16/block5_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block5_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block5_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block5_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
/vgg16_keypoint_detector/vgg16/block5_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
1vgg16_keypoint_detector/vgg16/block5_conv3/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block5_conv2/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
É
Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ÿ
2vgg16_keypoint_detector/vgg16/block5_conv3/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
/vgg16_keypoint_detector/vgg16/block5_conv3/ReluRelu;vgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
1vgg16_keypoint_detector/vgg16/block5_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
«
(vgg16_keypoint_detector/dropout/IdentityIdentity:vgg16_keypoint_detector/vgg16/block5_pool/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpQvgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ç
Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpSvgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:
*
dtype0
?vgg16_keypoint_detector/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
Gvgg16_keypoint_detector/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Å
Cvgg16_keypoint_detector/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative1vgg16_keypoint_detector/dropout/Identity:output:0Pvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
È
9vgg16_keypoint_detector/separable_conv2d/separable_conv2dConv2DLvgg16_keypoint_detector/separable_conv2d/separable_conv2d/depthwise:output:0Rvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
Ä
?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOpHvgg16_keypoint_detector_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
0vgg16_keypoint_detector/separable_conv2d/BiasAddBiasAddBvgg16_keypoint_detector/separable_conv2d/separable_conv2d:output:0Gvgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
-vgg16_keypoint_detector/separable_conv2d/ReluRelu9vgg16_keypoint_detector/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpSvgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0ê
Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpUvgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:

*
dtype0
Avgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
      
Ivgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ò
Evgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative;vgg16_keypoint_detector/separable_conv2d/Relu:activations:0Rvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
Î
;vgg16_keypoint_detector/separable_conv2d_1/separable_conv2dConv2DNvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/depthwise:output:0Tvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
È
Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
2vgg16_keypoint_detector/separable_conv2d_1/BiasAddBiasAddDvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d:output:0Ivgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
®
/vgg16_keypoint_detector/separable_conv2d_1/ReluRelu;vgg16_keypoint_detector/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

IdentityIdentity=vgg16_keypoint_detector/separable_conv2d_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Û
NoOpNoOp@^vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOpI^vgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOpK^vgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1B^vgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOpK^vgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOpM^vgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1B^vgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp2
Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOpHvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp2
Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_12
Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp2
Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOpJvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp2
Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_12
Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp2
Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp2
@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_2: 

_output_shapes
:


F__inference_block3_conv1_layer_call_and_return_conditional_losses_8076

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs


F__inference_block5_conv3_layer_call_and_return_conditional_losses_8256

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
&__inference_dropout_layer_call_fn_7885

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6621x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·

`
A__inference_dropout_layer_call_and_return_conditional_losses_6621

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
£
+__inference_block2_conv2_layer_call_fn_8035

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5644x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
¬
F
*__inference_block3_pool_layer_call_fn_8121

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_5547
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_block1_conv2_layer_call_and_return_conditional_losses_7996

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
 
_user_specified_nameinputs
õ
£
+__inference_block3_conv1_layer_call_fn_8065

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_5662x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
*
¢
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7071
input_2
tf_nn_bias_add_biasadd_bias$

vgg16_7002:@

vgg16_7004:@$

vgg16_7006:@@

vgg16_7008:@%

vgg16_7010:@

vgg16_7012:	&

vgg16_7014:

vgg16_7016:	&

vgg16_7018:

vgg16_7020:	&

vgg16_7022:

vgg16_7024:	&

vgg16_7026:

vgg16_7028:	&

vgg16_7030:

vgg16_7032:	&

vgg16_7034:

vgg16_7036:	&

vgg16_7038:

vgg16_7040:	&

vgg16_7042:

vgg16_7044:	&

vgg16_7046:

vgg16_7048:	&

vgg16_7050:

vgg16_7052:	0
separable_conv2d_7056:0
separable_conv2d_7058:
#
separable_conv2d_7060:
1
separable_conv2d_1_7063:
1
separable_conv2d_1_7065:

%
separable_conv2d_1_7067:

identity¢dropout/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall¢vgg16/StatefulPartitionedCall}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿè
&tf.__operators__.getitem/strided_sliceStridedSliceinput_25tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*

begin_mask*
ellipsis_mask*
end_mask«
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÐ
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0
vgg16_7002
vgg16_7004
vgg16_7006
vgg16_7008
vgg16_7010
vgg16_7012
vgg16_7014
vgg16_7016
vgg16_7018
vgg16_7020
vgg16_7022
vgg16_7024
vgg16_7026
vgg16_7028
vgg16_7030
vgg16_7032
vgg16_7034
vgg16_7036
vgg16_7038
vgg16_7040
vgg16_7042
vgg16_7044
vgg16_7046
vgg16_7048
vgg16_7050
vgg16_7052*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_6126í
dropout/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6621Í
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0separable_conv2d_7056separable_conv2d_7058separable_conv2d_7060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_6406à
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_7063separable_conv2d_1_7065separable_conv2d_1_7067*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_6435
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à
NoOpNoOp ^dropout/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_2: 

_output_shapes
:


F__inference_block4_conv3_layer_call_and_return_conditional_losses_5748

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block4_conv2_layer_call_and_return_conditional_losses_5731

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block5_conv2_layer_call_and_return_conditional_losses_5783

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
áU
ð
?__inference_vgg16_layer_call_and_return_conditional_losses_6312
input_1+
block1_conv1_6241:@
block1_conv1_6243:@+
block1_conv2_6246:@@
block1_conv2_6248:@,
block2_conv1_6252:@ 
block2_conv1_6254:	-
block2_conv2_6257: 
block2_conv2_6259:	-
block3_conv1_6263: 
block3_conv1_6265:	-
block3_conv2_6268: 
block3_conv2_6270:	-
block3_conv3_6273: 
block3_conv3_6275:	-
block4_conv1_6279: 
block4_conv1_6281:	-
block4_conv2_6284: 
block4_conv2_6286:	-
block4_conv3_6289: 
block4_conv3_6291:	-
block5_conv1_6295: 
block5_conv1_6297:	-
block5_conv2_6300: 
block5_conv2_6302:	-
block5_conv3_6305: 
block5_conv3_6307:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_6241block1_conv1_6243*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5592«
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6246block1_conv2_6248*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5609ë
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_5523¡
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_6252block2_conv1_6254*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5627ª
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6257block2_conv2_6259*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5644ì
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_5535¡
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_6263block3_conv1_6265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_5662ª
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6268block3_conv2_6270*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_5679ª
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_6273block3_conv3_6275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_5696ì
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_5547¡
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_6279block4_conv1_6281*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_5714ª
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_6284block4_conv2_6286*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_5731ª
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_6289block4_conv3_6291*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_5748ì
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_5559¡
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_6295block5_conv1_6297*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_5766ª
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_6300block5_conv2_6302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_5783ª
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_6305block5_conv3_6307*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_5800ì
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_5571|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
õ
£
+__inference_block3_conv3_layer_call_fn_8105

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_5696x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

a
E__inference_block3_pool_layer_call_and_return_conditional_losses_5547

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block4_conv2_layer_call_and_return_conditional_losses_8166

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_block3_conv3_layer_call_and_return_conditional_losses_5696

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
õ
£
+__inference_block4_conv2_layer_call_fn_8155

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_5731x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

J__inference_separable_conv2d_layer_call_and_return_conditional_losses_6406

inputsC
(separable_conv2d_readvariableop_resource:E
*separable_conv2d_readvariableop_1_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:
*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
ß
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block3_pool_layer_call_and_return_conditional_losses_8126

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¡	
6__inference_vgg16_keypoint_detector_layer_call_fn_6601
input_2
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@
	unknown_5:	%
	unknown_6:
	unknown_7:	%
	unknown_8:
	unknown_9:	&

unknown_10:

unknown_11:	&

unknown_12:

unknown_13:	&

unknown_14:

unknown_15:	&

unknown_16:

unknown_17:	&

unknown_18:

unknown_19:	&

unknown_20:

unknown_21:	&

unknown_22:

unknown_23:	&

unknown_24:

unknown_25:	%

unknown_26:%

unknown_27:


unknown_28:
$

unknown_29:
$

unknown_30:



unknown_31:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6532w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿàà:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_2: 

_output_shapes
:
ÝË
ü"
 __inference__traced_restore_8606
file_prefixM
2assignvariableop_separable_conv2d_depthwise_kernel:O
4assignvariableop_1_separable_conv2d_pointwise_kernel:
6
(assignvariableop_2_separable_conv2d_bias:
P
6assignvariableop_3_separable_conv2d_1_depthwise_kernel:
P
6assignvariableop_4_separable_conv2d_1_pointwise_kernel:

8
*assignvariableop_5_separable_conv2d_1_bias:
&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: A
'assignvariableop_11_block1_conv1_kernel:@3
%assignvariableop_12_block1_conv1_bias:@A
'assignvariableop_13_block1_conv2_kernel:@@3
%assignvariableop_14_block1_conv2_bias:@B
'assignvariableop_15_block2_conv1_kernel:@4
%assignvariableop_16_block2_conv1_bias:	C
'assignvariableop_17_block2_conv2_kernel:4
%assignvariableop_18_block2_conv2_bias:	C
'assignvariableop_19_block3_conv1_kernel:4
%assignvariableop_20_block3_conv1_bias:	C
'assignvariableop_21_block3_conv2_kernel:4
%assignvariableop_22_block3_conv2_bias:	C
'assignvariableop_23_block3_conv3_kernel:4
%assignvariableop_24_block3_conv3_bias:	C
'assignvariableop_25_block4_conv1_kernel:4
%assignvariableop_26_block4_conv1_bias:	C
'assignvariableop_27_block4_conv2_kernel:4
%assignvariableop_28_block4_conv2_bias:	C
'assignvariableop_29_block4_conv3_kernel:4
%assignvariableop_30_block4_conv3_bias:	C
'assignvariableop_31_block5_conv1_kernel:4
%assignvariableop_32_block5_conv1_bias:	C
'assignvariableop_33_block5_conv2_kernel:4
%assignvariableop_34_block5_conv2_bias:	C
'assignvariableop_35_block5_conv3_kernel:4
%assignvariableop_36_block5_conv3_bias:	#
assignvariableop_37_total: #
assignvariableop_38_count: W
<assignvariableop_39_adam_separable_conv2d_depthwise_kernel_m:W
<assignvariableop_40_adam_separable_conv2d_pointwise_kernel_m:
>
0assignvariableop_41_adam_separable_conv2d_bias_m:
X
>assignvariableop_42_adam_separable_conv2d_1_depthwise_kernel_m:
X
>assignvariableop_43_adam_separable_conv2d_1_pointwise_kernel_m:

@
2assignvariableop_44_adam_separable_conv2d_1_bias_m:
W
<assignvariableop_45_adam_separable_conv2d_depthwise_kernel_v:W
<assignvariableop_46_adam_separable_conv2d_pointwise_kernel_v:
>
0assignvariableop_47_adam_separable_conv2d_bias_v:
X
>assignvariableop_48_adam_separable_conv2d_1_depthwise_kernel_v:
X
>assignvariableop_49_adam_separable_conv2d_1_pointwise_kernel_v:

@
2assignvariableop_50_adam_separable_conv2d_1_bias_v:

identity_52¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ª
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*Ð
valueÆBÃ4B@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*æ
_output_shapesÓ
Ð::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp2assignvariableop_separable_conv2d_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_1AssignVariableOp4assignvariableop_1_separable_conv2d_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp(assignvariableop_2_separable_conv2d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_3AssignVariableOp6assignvariableop_3_separable_conv2d_1_depthwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_4AssignVariableOp6assignvariableop_4_separable_conv2d_1_pointwise_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_separable_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp'assignvariableop_11_block1_conv1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_block1_conv1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block1_conv2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block1_conv2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block2_conv1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block2_conv1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block2_conv2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block2_conv2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block3_conv1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block3_conv1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block3_conv2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block3_conv2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block3_conv3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block3_conv3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block4_conv1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block4_conv1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block4_conv2_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block4_conv2_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block4_conv3_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block4_conv3_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block5_conv1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block5_conv1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block5_conv2_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp%assignvariableop_34_block5_conv2_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block5_conv3_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_block5_conv3_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_39AssignVariableOp<assignvariableop_39_adam_separable_conv2d_depthwise_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_40AssignVariableOp<assignvariableop_40_adam_separable_conv2d_pointwise_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_separable_conv2d_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_separable_conv2d_1_depthwise_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_separable_conv2d_1_pointwise_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_separable_conv2d_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_separable_conv2d_depthwise_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_46AssignVariableOp<assignvariableop_46_adam_separable_conv2d_pointwise_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_47AssignVariableOp0assignvariableop_47_adam_separable_conv2d_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_separable_conv2d_1_depthwise_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_separable_conv2d_1_pointwise_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_separable_conv2d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ±	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
E
input_2:
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿààN
separable_conv2d_18
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:î

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
ü
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
 layer_with_weights-9
 layer-13
!layer-14
"layer_with_weights-10
"layer-15
#layer_with_weights-11
#layer-16
$layer_with_weights-12
$layer-17
%layer-18
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_network
¼
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
Û
3depthwise_kernel
4pointwise_kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
Û
<depthwise_kernel
=pointwise_kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
Ë
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_rate3mÉ4mÊ5mË<mÌ=mÍ>mÎ3vÏ4vÐ5vÑ<vÒ=vÓ>vÔ"
	optimizer

J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25
326
427
528
<29
=30
>31"
trackable_list_wrapper
J
30
41
52
<3
=4
>5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¦2£
6__inference_vgg16_keypoint_detector_layer_call_fn_6601
6__inference_vgg16_keypoint_detector_layer_call_fn_7148
6__inference_vgg16_keypoint_detector_layer_call_fn_7219
6__inference_vgg16_keypoint_detector_layer_call_fn_6915À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7350
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7488
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6993
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7071À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÊBÇ
__inference__wrapped_model_5514input_2"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
iserving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
"
_tf_keras_input_layer
»

Jkernel
Kbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Lkernel
Mbias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Nkernel
Obias
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Pkernel
Qbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Rkernel
Sbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Tkernel
Ubias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Vkernel
Wbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Xkernel
Ybias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Zkernel
[bias
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

\kernel
]bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

^kernel
_bias
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

`kernel
abias
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

bkernel
cbias
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"
_tf_keras_layer
æ
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Þ2Û
$__inference_vgg16_layer_call_fn_5863
$__inference_vgg16_layer_call_fn_7618
$__inference_vgg16_layer_call_fn_7675
$__inference_vgg16_layer_call_fn_6238À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
?__inference_vgg16_layer_call_and_return_conditional_losses_7775
?__inference_vgg16_layer_call_and_return_conditional_losses_7875
?__inference_vgg16_layer_call_and_return_conditional_losses_6312
?__inference_vgg16_layer_call_and_return_conditional_losses_6386À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
&__inference_dropout_layer_call_fn_7880
&__inference_dropout_layer_call_fn_7885´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_7890
A__inference_dropout_layer_call_and_return_conditional_losses_7902´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
<::2!separable_conv2d/depthwise_kernel
<::
2!separable_conv2d/pointwise_kernel
#:!
2separable_conv2d/bias
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_separable_conv2d_layer_call_fn_7913¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_7929¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
=:;
2#separable_conv2d_1/depthwise_kernel
=:;

2#separable_conv2d_1/pointwise_kernel
%:#
2separable_conv2d_1/bias
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_separable_conv2d_1_layer_call_fn_7940¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_7956¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
/:-2block2_conv2/kernel
 :2block2_conv2/bias
/:-2block3_conv1/kernel
 :2block3_conv1/bias
/:-2block3_conv2/kernel
 :2block3_conv2/bias
/:-2block3_conv3/kernel
 :2block3_conv3/bias
/:-2block4_conv1/kernel
 :2block4_conv1/bias
/:-2block4_conv2/kernel
 :2block4_conv2/bias
/:-2block4_conv3/kernel
 :2block4_conv3/bias
/:-2block5_conv1/kernel
 :2block5_conv1/bias
/:-2block5_conv2/kernel
 :2block5_conv2/bias
/:-2block5_conv3/kernel
 :2block5_conv3/bias
æ
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
(
ê0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÉBÆ
"__inference_signature_wrapper_7561input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block1_conv1_layer_call_fn_7965¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block1_conv1_layer_call_and_return_conditional_losses_7976¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block1_conv2_layer_call_fn_7985¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block1_conv2_layer_call_and_return_conditional_losses_7996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_block1_pool_layer_call_fn_8001¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_block1_pool_layer_call_and_return_conditional_losses_8006¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block2_conv1_layer_call_fn_8015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8026¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block2_conv2_layer_call_fn_8035¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8046¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_block2_pool_layer_call_fn_8051¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_block2_pool_layer_call_and_return_conditional_losses_8056¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block3_conv1_layer_call_fn_8065¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block3_conv2_layer_call_fn_8085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8096¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block3_conv3_layer_call_fn_8105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8116¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_block3_pool_layer_call_fn_8121¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_block3_pool_layer_call_and_return_conditional_losses_8126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block4_conv1_layer_call_fn_8135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8146¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block4_conv2_layer_call_fn_8155¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block4_conv2_layer_call_and_return_conditional_losses_8166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block4_conv3_layer_call_fn_8175¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block4_conv3_layer_call_and_return_conditional_losses_8186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_block4_pool_layer_call_fn_8191¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_block4_pool_layer_call_and_return_conditional_losses_8196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block5_conv1_layer_call_fn_8205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block5_conv1_layer_call_and_return_conditional_losses_8216¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block5_conv2_layer_call_fn_8225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block5_conv2_layer_call_and_return_conditional_losses_8236¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_block5_conv3_layer_call_fn_8245¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_block5_conv3_layer_call_and_return_conditional_losses_8256¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_block5_pool_layer_call_fn_8261¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_block5_pool_layer_call_and_return_conditional_losses_8266¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
æ
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
\18
]19
^20
_21
`22
a23
b24
c25"
trackable_list_wrapper
®
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Åtotal

Æcount
Ç	variables
È	keras_api"
_tf_keras_metric
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
Å0
Æ1"
trackable_list_wrapper
.
Ç	variables"
_generic_user_object
A:?2(Adam/separable_conv2d/depthwise_kernel/m
A:?
2(Adam/separable_conv2d/pointwise_kernel/m
(:&
2Adam/separable_conv2d/bias/m
B:@
2*Adam/separable_conv2d_1/depthwise_kernel/m
B:@

2*Adam/separable_conv2d_1/pointwise_kernel/m
*:(
2Adam/separable_conv2d_1/bias/m
A:?2(Adam/separable_conv2d/depthwise_kernel/v
A:?
2(Adam/separable_conv2d/pointwise_kernel/v
(:&
2Adam/separable_conv2d/bias/v
B:@
2*Adam/separable_conv2d_1/depthwise_kernel/v
B:@

2*Adam/separable_conv2d_1/pointwise_kernel/v
*:(
2Adam/separable_conv2d_1/bias/v
	J
ConstÕ
__inference__wrapped_model_5514±"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>:¢7
0¢-
+(
input_2ÿÿÿÿÿÿÿÿÿàà
ª "OªL
J
separable_conv2d_141
separable_conv2d_1ÿÿÿÿÿÿÿÿÿ
º
F__inference_block1_conv1_layer_call_and_return_conditional_losses_7976pJK9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà@
 
+__inference_block1_conv1_layer_call_fn_7965cJK9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª ""ÿÿÿÿÿÿÿÿÿàà@º
F__inference_block1_conv2_layer_call_and_return_conditional_losses_7996pLM9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà@
 
+__inference_block1_conv2_layer_call_fn_7985cLM9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà@
ª ""ÿÿÿÿÿÿÿÿÿàà@è
E__inference_block1_pool_layer_call_and_return_conditional_losses_8006R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block1_pool_layer_call_fn_8001R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8026mNO7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
+__inference_block2_conv1_layer_call_fn_8015`NO7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp@
ª "!ÿÿÿÿÿÿÿÿÿpp¸
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8046nPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
+__inference_block2_conv2_layer_call_fn_8035aPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª "!ÿÿÿÿÿÿÿÿÿppè
E__inference_block2_pool_layer_call_and_return_conditional_losses_8056R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block2_pool_layer_call_fn_8051R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8076nRS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
+__inference_block3_conv1_layer_call_fn_8065aRS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88¸
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8096nTU8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
+__inference_block3_conv2_layer_call_fn_8085aTU8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88¸
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8116nVW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
+__inference_block3_conv3_layer_call_fn_8105aVW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88è
E__inference_block3_pool_layer_call_and_return_conditional_losses_8126R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block3_pool_layer_call_fn_8121R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8146nXY8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_block4_conv1_layer_call_fn_8135aXY8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_block4_conv2_layer_call_and_return_conditional_losses_8166nZ[8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_block4_conv2_layer_call_fn_8155aZ[8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_block4_conv3_layer_call_and_return_conditional_losses_8186n\]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_block4_conv3_layer_call_fn_8175a\]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿè
E__inference_block4_pool_layer_call_and_return_conditional_losses_8196R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block4_pool_layer_call_fn_8191R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
F__inference_block5_conv1_layer_call_and_return_conditional_losses_8216n^_8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_block5_conv1_layer_call_fn_8205a^_8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_block5_conv2_layer_call_and_return_conditional_losses_8236n`a8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_block5_conv2_layer_call_fn_8225a`a8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_block5_conv3_layer_call_and_return_conditional_losses_8256nbc8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_block5_conv3_layer_call_fn_8245abc8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿè
E__inference_block5_pool_layer_call_and_return_conditional_losses_8266R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block5_pool_layer_call_fn_8261R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
A__inference_dropout_layer_call_and_return_conditional_losses_7890n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ³
A__inference_dropout_layer_call_and_return_conditional_losses_7902n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_dropout_layer_call_fn_7880a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ
&__inference_dropout_layer_call_fn_7885a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿâ
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_7956<=>I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 º
1__inference_separable_conv2d_1_layer_call_fn_7940<=>I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
á
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_7929345J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 ¹
/__inference_separable_conv2d_layer_call_fn_7913345J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ã
"__inference_signature_wrapper_7561¼"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>E¢B
¢ 
;ª8
6
input_2+(
input_2ÿÿÿÿÿÿÿÿÿàà"OªL
J
separable_conv2d_141
separable_conv2d_1ÿÿÿÿÿÿÿÿÿ
í
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_6993"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>B¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 í
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7071"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>B¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿàà
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 ì
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7350"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 ì
Q__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_7488"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 Å
6__inference_vgg16_keypoint_detector_layer_call_fn_6601"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>B¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿàà
p 

 
ª " ÿÿÿÿÿÿÿÿÿ
Å
6__inference_vgg16_keypoint_detector_layer_call_fn_6915"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>B¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿàà
p

 
ª " ÿÿÿÿÿÿÿÿÿ
Ä
6__inference_vgg16_keypoint_detector_layer_call_fn_7148"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª " ÿÿÿÿÿÿÿÿÿ
Ä
6__inference_vgg16_keypoint_detector_layer_call_fn_7219"ÕJKLMNOPQRSTUVWXYZ[\]^_`abc345<=>A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª " ÿÿÿÿÿÿÿÿÿ
Ô
?__inference_vgg16_layer_call_and_return_conditional_losses_6312JKLMNOPQRSTUVWXYZ[\]^_`abcB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ô
?__inference_vgg16_layer_call_and_return_conditional_losses_6386JKLMNOPQRSTUVWXYZ[\]^_`abcB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ó
?__inference_vgg16_layer_call_and_return_conditional_losses_7775JKLMNOPQRSTUVWXYZ[\]^_`abcA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ó
?__inference_vgg16_layer_call_and_return_conditional_losses_7875JKLMNOPQRSTUVWXYZ[\]^_`abcA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¬
$__inference_vgg16_layer_call_fn_5863JKLMNOPQRSTUVWXYZ[\]^_`abcB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ¬
$__inference_vgg16_layer_call_fn_6238JKLMNOPQRSTUVWXYZ[\]^_`abcB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p

 
ª "!ÿÿÿÿÿÿÿÿÿ«
$__inference_vgg16_layer_call_fn_7618JKLMNOPQRSTUVWXYZ[\]^_`abcA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ«
$__inference_vgg16_layer_call_fn_7675JKLMNOPQRSTUVWXYZ[\]^_`abcA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "!ÿÿÿÿÿÿÿÿÿ
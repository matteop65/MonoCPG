??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
?
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!separable_conv2d/depthwise_kernel
?
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*'
_output_shapes
:?*
dtype0
?
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!separable_conv2d/pointwise_kernel
?
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*'
_output_shapes
:?*
dtype0
?
separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameseparable_conv2d/bias
{
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes
:*
dtype0
?
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_1/depthwise_kernel
?
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
:*
dtype0
?
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_1/pointwise_kernel
?
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*&
_output_shapes
:*
dtype0
?
separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameseparable_conv2d_1/bias

+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes
:*
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
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
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
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
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
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0
?
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv2/kernel
?
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:?*
dtype0
?
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv3/kernel
?
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:?*
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
?
(Adam/separable_conv2d/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/m
?
<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/m*'
_output_shapes
:?*
dtype0
?
(Adam/separable_conv2d/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/m
?
<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/m*'
_output_shapes
:?*
dtype0
?
Adam/separable_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/separable_conv2d/bias/m
?
0Adam/separable_conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/m*
_output_shapes
:*
dtype0
?
*Adam/separable_conv2d_1/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/m
?
>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/m*&
_output_shapes
:*
dtype0
?
*Adam/separable_conv2d_1/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/m
?
>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/m*&
_output_shapes
:*
dtype0
?
Adam/separable_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_1/bias/m
?
2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
(Adam/separable_conv2d/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/v
?
<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/v*'
_output_shapes
:?*
dtype0
?
(Adam/separable_conv2d/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/v
?
<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/v*'
_output_shapes
:?*
dtype0
?
Adam/separable_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/separable_conv2d/bias/v
?
0Adam/separable_conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/v*
_output_shapes
:*
dtype0
?
*Adam/separable_conv2d_1/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/v
?
>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/v*&
_output_shapes
:*
dtype0
?
*Adam/separable_conv2d_1/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/v
?
>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/v*&
_output_shapes
:*
dtype0
?
Adam/separable_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_1/bias/v
?
2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/v*
_output_shapes
:*
dtype0
Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"????َ??)\??

NoOpNoOp
?s
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?s
value?sB?s B?s
?
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

regularization_losses
trainable_variables
	keras_api

signatures
 

	keras_api

	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
 layer_with_weights-11
 layer-16
!layer_with_weights-12
!layer-17
"layer-18
#	variables
$regularization_losses
%trainable_variables
&	keras_api
R
'	variables
(regularization_losses
)trainable_variables
*	keras_api
?
+depthwise_kernel
,pointwise_kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
?
2depthwise_kernel
3pointwise_kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
?
9iter

:beta_1

;beta_2
	<decay
=learning_rate+m?,m?-m?2m?3m?4m?+v?,v?-v?2v?3v?4v?
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
+26
,27
-28
229
330
431
 
*
+0
,1
-2
23
34
45
?
		variables

regularization_losses
Xnon_trainable_variables
trainable_variables
Ymetrics
Zlayer_regularization_losses
[layer_metrics

\layers
 
 
 
 
h

>kernel
?bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
h

@kernel
Abias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
h

Bkernel
Cbias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
h

Dkernel
Ebias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
R
q	variables
rregularization_losses
strainable_variables
t	keras_api
h

Fkernel
Gbias
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
h

Hkernel
Ibias
y	variables
zregularization_losses
{trainable_variables
|	keras_api
i

Jkernel
Kbias
}	variables
~regularization_losses
trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Lkernel
Mbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Nkernel
Obias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Pkernel
Qbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Rkernel
Sbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Tkernel
Ubias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Vkernel
Wbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
 
 
?
#	variables
$regularization_losses
?non_trainable_variables
%trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
 
 
 
?
'	variables
(regularization_losses
?non_trainable_variables
)trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
wu
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
-2
 

+0
,1
-2
?
.	variables
/regularization_losses
?non_trainable_variables
0trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
yw
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
42
 

20
31
42
?
5	variables
6regularization_losses
?non_trainable_variables
7trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25

?0
 
 
1
0
1
2
3
4
5
6

>0
?1
 
 
?
]	variables
^regularization_losses
?non_trainable_variables
_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

@0
A1
 
 
?
a	variables
bregularization_losses
?non_trainable_variables
ctrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
 
 
 
?
e	variables
fregularization_losses
?non_trainable_variables
gtrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

B0
C1
 
 
?
i	variables
jregularization_losses
?non_trainable_variables
ktrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

D0
E1
 
 
?
m	variables
nregularization_losses
?non_trainable_variables
otrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
 
 
 
?
q	variables
rregularization_losses
?non_trainable_variables
strainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

F0
G1
 
 
?
u	variables
vregularization_losses
?non_trainable_variables
wtrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

H0
I1
 
 
?
y	variables
zregularization_losses
?non_trainable_variables
{trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

J0
K1
 
 
?
}	variables
~regularization_losses
?non_trainable_variables
trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
 
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

L0
M1
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

N0
O1
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

P0
Q1
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
 
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

R0
S1
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

T0
U1
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

V0
W1
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
 
 
 
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
 
 
 
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api

>0
?1
 
 
 
 

@0
A1
 
 
 
 
 
 
 
 
 

B0
C1
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 

F0
G1
 
 
 
 

H0
I1
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 

N0
O1
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 

R0
S1
 
 
 
 

T0
U1
 
 
 
 

V0
W1
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/m\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/m\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/separable_conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/separable_conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/v\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/v\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/separable_conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/separable_conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2Constblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 !*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_95779
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_96968
?
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
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_97131??
?
C
'__inference_dropout_layer_call_fn_96526

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_951442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95622
input_2
tf_nn_bias_add_biasadd_bias%
vgg16_95553:@
vgg16_95555:@%
vgg16_95557:@@
vgg16_95559:@&
vgg16_95561:@?
vgg16_95563:	?'
vgg16_95565:??
vgg16_95567:	?'
vgg16_95569:??
vgg16_95571:	?'
vgg16_95573:??
vgg16_95575:	?'
vgg16_95577:??
vgg16_95579:	?'
vgg16_95581:??
vgg16_95583:	?'
vgg16_95585:??
vgg16_95587:	?'
vgg16_95589:??
vgg16_95591:	?'
vgg16_95593:??
vgg16_95595:	?'
vgg16_95597:??
vgg16_95599:	?'
vgg16_95601:??
vgg16_95603:	?1
separable_conv2d_95607:?1
separable_conv2d_95609:?$
separable_conv2d_95611:2
separable_conv2d_1_95614:2
separable_conv2d_1_95616:&
separable_conv2d_1_95618:
identity??(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinput_25tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2(
&tf.__operators__.getitem/strided_slice?
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????2
tf.nn.bias_add/BiasAdd?
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0vgg16_95553vgg16_95555vgg16_95557vgg16_95559vgg16_95561vgg16_95563vgg16_95565vgg16_95567vgg16_95569vgg16_95571vgg16_95573vgg16_95575vgg16_95577vgg16_95579vgg16_95581vgg16_95583vgg16_95585vgg16_95587vgg16_95589vgg16_95591vgg16_95593vgg16_95595vgg16_95597vgg16_95599vgg16_95601vgg16_95603*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_944372
vgg16/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_951442
dropout/PartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0separable_conv2d_95607separable_conv2d_95609separable_conv2d_95611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_950322*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95614separable_conv2d_1_95616separable_conv2d_1_95618*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_950612,
*separable_conv2d_1/StatefulPartitionedCall?
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2: 

_output_shapes
:
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_96622

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?[
?
@__inference_vgg16_layer_call_and_return_conditional_losses_94437

inputs,
block1_conv1_94222:@ 
block1_conv1_94224:@,
block1_conv2_94239:@@ 
block1_conv2_94241:@-
block2_conv1_94257:@?!
block2_conv1_94259:	?.
block2_conv2_94274:??!
block2_conv2_94276:	?.
block3_conv1_94292:??!
block3_conv1_94294:	?.
block3_conv2_94309:??!
block3_conv2_94311:	?.
block3_conv3_94326:??!
block3_conv3_94328:	?.
block4_conv1_94344:??!
block4_conv1_94346:	?.
block4_conv2_94361:??!
block4_conv2_94363:	?.
block4_conv3_94378:??!
block4_conv3_94380:	?.
block5_conv1_94396:??!
block5_conv1_94398:	?.
block5_conv2_94413:??!
block5_conv2_94415:	?.
block5_conv3_94430:??!
block5_conv3_94432:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_94222block1_conv1_94224*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_942212&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_94239block1_conv2_94241*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_942382&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_941492
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_94257block2_conv1_94259*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_942562&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_94274block2_conv2_94276*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_942732&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_941612
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_94292block3_conv1_94294*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_942912&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_94309block3_conv2_94311*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_943082&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_94326block3_conv3_94328*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_943252&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_941732
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_94344block4_conv1_94346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_943432&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_94361block4_conv2_94363*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_943602&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_94378block4_conv3_94380*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_943772&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_941852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_94396block5_conv1_94398*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_943952&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_94413block5_conv2_94415*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_944122&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_94430block5_conv3_94432*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_944292&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_941972
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_96722

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_94325

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?-
?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95404

inputs
tf_nn_bias_add_biasadd_bias%
vgg16_95335:@
vgg16_95337:@%
vgg16_95339:@@
vgg16_95341:@&
vgg16_95343:@?
vgg16_95345:	?'
vgg16_95347:??
vgg16_95349:	?'
vgg16_95351:??
vgg16_95353:	?'
vgg16_95355:??
vgg16_95357:	?'
vgg16_95359:??
vgg16_95361:	?'
vgg16_95363:??
vgg16_95365:	?'
vgg16_95367:??
vgg16_95369:	?'
vgg16_95371:??
vgg16_95373:	?'
vgg16_95375:??
vgg16_95377:	?'
vgg16_95379:??
vgg16_95381:	?'
vgg16_95383:??
vgg16_95385:	?1
separable_conv2d_95389:?1
separable_conv2d_95391:?$
separable_conv2d_95393:2
separable_conv2d_1_95396:2
separable_conv2d_1_95398:&
separable_conv2d_1_95400:
identity??dropout/StatefulPartitionedCall?(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2(
&tf.__operators__.getitem/strided_slice?
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????2
tf.nn.bias_add/BiasAdd?
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0vgg16_95335vgg16_95337vgg16_95339vgg16_95341vgg16_95343vgg16_95345vgg16_95347vgg16_95349vgg16_95351vgg16_95353vgg16_95355vgg16_95357vgg16_95359vgg16_95361vgg16_95363vgg16_95365vgg16_95367vgg16_95369vgg16_95371vgg16_95373vgg16_95375vgg16_95377vgg16_95379vgg16_95381vgg16_95383vgg16_95385*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_947552
vgg16/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_952502!
dropout/StatefulPartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0separable_conv2d_95389separable_conv2d_95391separable_conv2d_95393*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_950322*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95396separable_conv2d_1_95398separable_conv2d_1_95400*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_950612,
*separable_conv2d_1/StatefulPartitionedCall?
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?[
?
@__inference_vgg16_layer_call_and_return_conditional_losses_95015
input_1,
block1_conv1_94944:@ 
block1_conv1_94946:@,
block1_conv2_94949:@@ 
block1_conv2_94951:@-
block2_conv1_94955:@?!
block2_conv1_94957:	?.
block2_conv2_94960:??!
block2_conv2_94962:	?.
block3_conv1_94966:??!
block3_conv1_94968:	?.
block3_conv2_94971:??!
block3_conv2_94973:	?.
block3_conv3_94976:??!
block3_conv3_94978:	?.
block4_conv1_94982:??!
block4_conv1_94984:	?.
block4_conv2_94987:??!
block4_conv2_94989:	?.
block4_conv3_94992:??!
block4_conv3_94994:	?.
block5_conv1_94998:??!
block5_conv1_95000:	?.
block5_conv2_95003:??!
block5_conv2_95005:	?.
block5_conv3_95008:??!
block5_conv3_95010:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_94944block1_conv1_94946*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_942212&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_94949block1_conv2_94951*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_942382&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_941492
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_94955block2_conv1_94957*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_942562&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_94960block2_conv2_94962*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_942732&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_941612
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_94966block3_conv1_94968*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_942912&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_94971block3_conv2_94973*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_943082&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_94976block3_conv3_94978*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_943252&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_941732
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_94982block4_conv1_94984*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_943432&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_94987block4_conv2_94989*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_943602&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_94992block4_conv3_94994*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_943772&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_941852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_94998block5_conv1_95000*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_943952&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_95003block5_conv2_95005*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_944122&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_95008block5_conv3_95010*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_944292&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_941972
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:???????????
!
_user_specified_name	input_1
?
?
,__inference_block4_conv3_layer_call_fn_96731

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_943772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block3_conv3_layer_call_fn_96671

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_943252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
??
?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_96048

inputs
tf_nn_bias_add_biasadd_biasK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block2_conv2_conv2d_readvariableop_resource:??A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv1_conv2d_readvariableop_resource:??A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv2_conv2d_readvariableop_resource:??A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv3_conv2d_readvariableop_resource:??A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv1_conv2d_readvariableop_resource:??A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv2_conv2d_readvariableop_resource:??A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv3_conv2d_readvariableop_resource:??A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv1_conv2d_readvariableop_resource:??A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv2_conv2d_readvariableop_resource:??A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv3_conv2d_readvariableop_resource:??A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	?T
9separable_conv2d_separable_conv2d_readvariableop_resource:?V
;separable_conv2d_separable_conv2d_readvariableop_1_resource:?>
0separable_conv2d_biasadd_readvariableop_resource:U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_1_biasadd_readvariableop_resource:
identity??'separable_conv2d/BiasAdd/ReadVariableOp?0separable_conv2d/separable_conv2d/ReadVariableOp?2separable_conv2d/separable_conv2d/ReadVariableOp_1?)separable_conv2d_1/BiasAdd/ReadVariableOp?2separable_conv2d_1/separable_conv2d/ReadVariableOp?4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2(
&tf.__operators__.getitem/strided_slice?
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????2
tf.nn.bias_add/BiasAdd?
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg16/block1_conv1/Conv2D/ReadVariableOp?
vgg16/block1_conv1/Conv2DConv2Dtf.nn.bias_add/BiasAdd:output:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv1/Conv2D?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv1/BiasAdd/ReadVariableOp?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/BiasAdd?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/Relu?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg16/block1_conv2/Conv2D/ReadVariableOp?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv2/Conv2D?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv2/BiasAdd/ReadVariableOp?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/BiasAdd?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/Relu?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
vgg16/block1_pool/MaxPool?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(vgg16/block2_conv1/Conv2D/ReadVariableOp?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv1/Conv2D?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv1/BiasAdd/ReadVariableOp?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/BiasAdd?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/Relu?
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block2_conv2/Conv2D/ReadVariableOp?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv2/Conv2D?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv2/BiasAdd/ReadVariableOp?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/BiasAdd?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/Relu?
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
vgg16/block2_pool/MaxPool?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv1/Conv2D/ReadVariableOp?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv1/Conv2D?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv1/BiasAdd/ReadVariableOp?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/BiasAdd?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/Relu?
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv2/Conv2D/ReadVariableOp?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv2/Conv2D?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv2/BiasAdd/ReadVariableOp?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/BiasAdd?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/Relu?
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv3/Conv2D/ReadVariableOp?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv3/Conv2D?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv3/BiasAdd/ReadVariableOp?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/BiasAdd?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/Relu?
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block3_pool/MaxPool?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv1/Conv2D/ReadVariableOp?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv1/Conv2D?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv1/BiasAdd/ReadVariableOp?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/BiasAdd?
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/Relu?
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv2/Conv2D/ReadVariableOp?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv2/Conv2D?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv2/BiasAdd/ReadVariableOp?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/BiasAdd?
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/Relu?
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv3/Conv2D/ReadVariableOp?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv3/Conv2D?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv3/BiasAdd/ReadVariableOp?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/BiasAdd?
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/Relu?
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block4_pool/MaxPool?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv1/Conv2D/ReadVariableOp?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv1/Conv2D?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv1/BiasAdd/ReadVariableOp?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/BiasAdd?
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/Relu?
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv2/Conv2D/ReadVariableOp?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv2/Conv2D?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv2/BiasAdd/ReadVariableOp?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/BiasAdd?
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/Relu?
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv3/Conv2D/ReadVariableOp?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv3/Conv2D?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv3/BiasAdd/ReadVariableOp?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/BiasAdd?
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/Relu?
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block5_pool/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMul"vgg16/block5_pool/MaxPool:output:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/Mul?
dropout/dropout/ShapeShape"vgg16/block5_pool/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/Mul_1?
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOp?
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:?*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1?
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'separable_conv2d/separable_conv2d/Shape?
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rate?
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativedropout/dropout/Mul_1:z:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2-
+separable_conv2d/separable_conv2d/depthwise?
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d?
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp?
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
separable_conv2d/BiasAdd?
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
separable_conv2d/Relu?
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOp?
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_1/separable_conv2d/Shape?
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rate?
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise?
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d?
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOp?
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
separable_conv2d_1/BiasAdd?
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
separable_conv2d_1/Relu?
IdentityIdentity%separable_conv2d_1/Relu:activations:0(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
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
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
G
+__inference_block2_pool_layer_call_fn_94167

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_941612
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_94149

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_block5_pool_layer_call_fn_94203

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_941972
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_94221

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_block1_conv2_layer_call_fn_96571

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_942382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_95144

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95161

inputs
tf_nn_bias_add_biasadd_bias%
vgg16_95086:@
vgg16_95088:@%
vgg16_95090:@@
vgg16_95092:@&
vgg16_95094:@?
vgg16_95096:	?'
vgg16_95098:??
vgg16_95100:	?'
vgg16_95102:??
vgg16_95104:	?'
vgg16_95106:??
vgg16_95108:	?'
vgg16_95110:??
vgg16_95112:	?'
vgg16_95114:??
vgg16_95116:	?'
vgg16_95118:??
vgg16_95120:	?'
vgg16_95122:??
vgg16_95124:	?'
vgg16_95126:??
vgg16_95128:	?'
vgg16_95130:??
vgg16_95132:	?'
vgg16_95134:??
vgg16_95136:	?1
separable_conv2d_95146:?1
separable_conv2d_95148:?$
separable_conv2d_95150:2
separable_conv2d_1_95153:2
separable_conv2d_1_95155:&
separable_conv2d_1_95157:
identity??(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2(
&tf.__operators__.getitem/strided_slice?
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????2
tf.nn.bias_add/BiasAdd?
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0vgg16_95086vgg16_95088vgg16_95090vgg16_95092vgg16_95094vgg16_95096vgg16_95098vgg16_95100vgg16_95102vgg16_95104vgg16_95106vgg16_95108vgg16_95110vgg16_95112vgg16_95114vgg16_95116vgg16_95118vgg16_95120vgg16_95122vgg16_95124vgg16_95126vgg16_95128vgg16_95130vgg16_95132vgg16_95134vgg16_95136*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_944372
vgg16/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_951442
dropout/PartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0separable_conv2d_95146separable_conv2d_95148separable_conv2d_95150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_950322*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95153separable_conv2d_1_95155separable_conv2d_1_95157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_950612,
*separable_conv2d_1/StatefulPartitionedCall?
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?[
?
@__inference_vgg16_layer_call_and_return_conditional_losses_94755

inputs,
block1_conv1_94684:@ 
block1_conv1_94686:@,
block1_conv2_94689:@@ 
block1_conv2_94691:@-
block2_conv1_94695:@?!
block2_conv1_94697:	?.
block2_conv2_94700:??!
block2_conv2_94702:	?.
block3_conv1_94706:??!
block3_conv1_94708:	?.
block3_conv2_94711:??!
block3_conv2_94713:	?.
block3_conv3_94716:??!
block3_conv3_94718:	?.
block4_conv1_94722:??!
block4_conv1_94724:	?.
block4_conv2_94727:??!
block4_conv2_94729:	?.
block4_conv3_94732:??!
block4_conv3_94734:	?.
block5_conv1_94738:??!
block5_conv1_94740:	?.
block5_conv2_94743:??!
block5_conv2_94745:	?.
block5_conv3_94748:??!
block5_conv3_94750:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_94684block1_conv1_94686*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_942212&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_94689block1_conv2_94691*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_942382&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_941492
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_94695block2_conv1_94697*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_942562&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_94700block2_conv2_94702*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_942732&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_941612
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_94706block3_conv1_94708*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_942912&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_94711block3_conv2_94713*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_943082&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_94716block3_conv3_94718*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_943252&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_941732
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_94722block4_conv1_94724*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_943432&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_94727block4_conv2_94729*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_943602&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_94732block4_conv3_94734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_943772&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_941852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_94738block5_conv1_94740*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_943952&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_94743block5_conv2_94745*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_944122&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_94748block5_conv3_94750*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_944292&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_941972
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:???????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_96509

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
ɖ
?
@__inference_vgg16_layer_call_and_return_conditional_losses_96290

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv3/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:???????????
 
_user_specified_nameinputs
?
G
+__inference_block3_pool_layer_call_fn_94179

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_941732
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
0__inference_separable_conv2d_layer_call_fn_95044

inputs"
unknown:?$
	unknown_0:?
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_950322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,????????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?[
?
@__inference_vgg16_layer_call_and_return_conditional_losses_94941
input_1,
block1_conv1_94870:@ 
block1_conv1_94872:@,
block1_conv2_94875:@@ 
block1_conv2_94877:@-
block2_conv1_94881:@?!
block2_conv1_94883:	?.
block2_conv2_94886:??!
block2_conv2_94888:	?.
block3_conv1_94892:??!
block3_conv1_94894:	?.
block3_conv2_94897:??!
block3_conv2_94899:	?.
block3_conv3_94902:??!
block3_conv3_94904:	?.
block4_conv1_94908:??!
block4_conv1_94910:	?.
block4_conv2_94913:??!
block4_conv2_94915:	?.
block4_conv3_94918:??!
block4_conv3_94920:	?.
block5_conv1_94924:??!
block5_conv1_94926:	?.
block5_conv2_94929:??!
block5_conv2_94931:	?.
block5_conv3_94934:??!
block5_conv3_94936:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_94870block1_conv1_94872*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_942212&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_94875block1_conv2_94877*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_942382&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_941492
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_94881block2_conv1_94883*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_942562&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_94886block2_conv2_94888*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_942732&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_941612
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_94892block3_conv1_94894*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_942912&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_94897block3_conv2_94899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_943082&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_94902block3_conv3_94904*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_943252&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_941732
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_94908block4_conv1_94910*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_943432&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_94913block4_conv2_94915*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_943602&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_94918block4_conv3_94920*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_943772&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_941852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_94924block5_conv1_94926*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_943952&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_94929block5_conv2_94931*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_944122&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_94934block5_conv3_94936*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_944292&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_941972
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:???????????
!
_user_specified_name	input_1
?
?
,__inference_block2_conv1_layer_call_fn_96591

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_942562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_96682

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
7__inference_vgg16_keypoint_detector_layer_call_fn_96190

inputs
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?%

unknown_26:?%

unknown_27:?

unknown_28:$

unknown_29:$

unknown_30:

unknown_31:
identity??StatefulPartitionedCall?
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
:?????????*B
_read_only_resource_inputs$
" 	
 !*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_954042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
?
%__inference_vgg16_layer_call_fn_94867
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
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
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_947552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
??
?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95910

inputs
tf_nn_bias_add_biasadd_biasK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block2_conv2_conv2d_readvariableop_resource:??A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv1_conv2d_readvariableop_resource:??A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv2_conv2d_readvariableop_resource:??A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv3_conv2d_readvariableop_resource:??A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv1_conv2d_readvariableop_resource:??A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv2_conv2d_readvariableop_resource:??A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv3_conv2d_readvariableop_resource:??A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv1_conv2d_readvariableop_resource:??A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv2_conv2d_readvariableop_resource:??A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv3_conv2d_readvariableop_resource:??A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	?T
9separable_conv2d_separable_conv2d_readvariableop_resource:?V
;separable_conv2d_separable_conv2d_readvariableop_1_resource:?>
0separable_conv2d_biasadd_readvariableop_resource:U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_1_biasadd_readvariableop_resource:
identity??'separable_conv2d/BiasAdd/ReadVariableOp?0separable_conv2d/separable_conv2d/ReadVariableOp?2separable_conv2d/separable_conv2d/ReadVariableOp_1?)separable_conv2d_1/BiasAdd/ReadVariableOp?2separable_conv2d_1/separable_conv2d/ReadVariableOp?4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2(
&tf.__operators__.getitem/strided_slice?
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????2
tf.nn.bias_add/BiasAdd?
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg16/block1_conv1/Conv2D/ReadVariableOp?
vgg16/block1_conv1/Conv2DConv2Dtf.nn.bias_add/BiasAdd:output:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv1/Conv2D?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv1/BiasAdd/ReadVariableOp?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/BiasAdd?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/Relu?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg16/block1_conv2/Conv2D/ReadVariableOp?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv2/Conv2D?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv2/BiasAdd/ReadVariableOp?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/BiasAdd?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/Relu?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
vgg16/block1_pool/MaxPool?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(vgg16/block2_conv1/Conv2D/ReadVariableOp?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv1/Conv2D?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv1/BiasAdd/ReadVariableOp?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/BiasAdd?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/Relu?
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block2_conv2/Conv2D/ReadVariableOp?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv2/Conv2D?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv2/BiasAdd/ReadVariableOp?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/BiasAdd?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/Relu?
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
vgg16/block2_pool/MaxPool?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv1/Conv2D/ReadVariableOp?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv1/Conv2D?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv1/BiasAdd/ReadVariableOp?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/BiasAdd?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/Relu?
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv2/Conv2D/ReadVariableOp?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv2/Conv2D?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv2/BiasAdd/ReadVariableOp?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/BiasAdd?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/Relu?
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv3/Conv2D/ReadVariableOp?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv3/Conv2D?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv3/BiasAdd/ReadVariableOp?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/BiasAdd?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/Relu?
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block3_pool/MaxPool?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv1/Conv2D/ReadVariableOp?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv1/Conv2D?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv1/BiasAdd/ReadVariableOp?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/BiasAdd?
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/Relu?
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv2/Conv2D/ReadVariableOp?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv2/Conv2D?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv2/BiasAdd/ReadVariableOp?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/BiasAdd?
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/Relu?
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv3/Conv2D/ReadVariableOp?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv3/Conv2D?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv3/BiasAdd/ReadVariableOp?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/BiasAdd?
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/Relu?
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block4_pool/MaxPool?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv1/Conv2D/ReadVariableOp?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv1/Conv2D?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv1/BiasAdd/ReadVariableOp?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/BiasAdd?
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/Relu?
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv2/Conv2D/ReadVariableOp?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv2/Conv2D?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv2/BiasAdd/ReadVariableOp?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/BiasAdd?
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/Relu?
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv3/Conv2D/ReadVariableOp?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv3/Conv2D?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv3/BiasAdd/ReadVariableOp?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/BiasAdd?
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/Relu?
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block5_pool/MaxPool?
dropout/IdentityIdentity"vgg16/block5_pool/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout/Identity?
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOp?
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:?*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1?
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'separable_conv2d/separable_conv2d/Shape?
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rate?
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativedropout/Identity:output:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2-
+separable_conv2d/separable_conv2d/depthwise?
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d?
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp?
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
separable_conv2d/BiasAdd?
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
separable_conv2d/Relu?
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOp?
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_1/separable_conv2d/Shape?
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rate?
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise?
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d?
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOp?
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
separable_conv2d_1/BiasAdd?
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
separable_conv2d_1/Relu?
IdentityIdentity%separable_conv2d_1/Relu:activations:0(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
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
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_94197

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95032

inputsC
(separable_conv2d_readvariableop_resource:?E
*separable_conv2d_readvariableop_1_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:?*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,????????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
ɖ
?
@__inference_vgg16_layer_call_and_return_conditional_losses_96390

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv3/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:???????????
 
_user_specified_nameinputs
?-
?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95700
input_2
tf_nn_bias_add_biasadd_bias%
vgg16_95631:@
vgg16_95633:@%
vgg16_95635:@@
vgg16_95637:@&
vgg16_95639:@?
vgg16_95641:	?'
vgg16_95643:??
vgg16_95645:	?'
vgg16_95647:??
vgg16_95649:	?'
vgg16_95651:??
vgg16_95653:	?'
vgg16_95655:??
vgg16_95657:	?'
vgg16_95659:??
vgg16_95661:	?'
vgg16_95663:??
vgg16_95665:	?'
vgg16_95667:??
vgg16_95669:	?'
vgg16_95671:??
vgg16_95673:	?'
vgg16_95675:??
vgg16_95677:	?'
vgg16_95679:??
vgg16_95681:	?1
separable_conv2d_95685:?1
separable_conv2d_95687:?$
separable_conv2d_95689:2
separable_conv2d_1_95692:2
separable_conv2d_1_95694:&
separable_conv2d_1_95696:
identity??dropout/StatefulPartitionedCall?(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinput_25tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2(
&tf.__operators__.getitem/strided_slice?
tf.nn.bias_add/BiasAddBiasAdd/tf.__operators__.getitem/strided_slice:output:0tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????2
tf.nn.bias_add/BiasAdd?
vgg16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.bias_add/BiasAdd:output:0vgg16_95631vgg16_95633vgg16_95635vgg16_95637vgg16_95639vgg16_95641vgg16_95643vgg16_95645vgg16_95647vgg16_95649vgg16_95651vgg16_95653vgg16_95655vgg16_95657vgg16_95659vgg16_95661vgg16_95663vgg16_95665vgg16_95667vgg16_95669vgg16_95671vgg16_95673vgg16_95675vgg16_95677vgg16_95679vgg16_95681*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_947552
vgg16/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_952502!
dropout/StatefulPartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0separable_conv2d_95685separable_conv2d_95687separable_conv2d_95689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_950322*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95692separable_conv2d_1_95694separable_conv2d_1_95696*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_950612,
*separable_conv2d_1/StatefulPartitionedCall?
IdentityIdentity3separable_conv2d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2: 

_output_shapes
:
?
?
,__inference_block5_conv1_layer_call_fn_96751

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_943952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block1_pool_layer_call_fn_94155

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_941492
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv2_layer_call_fn_96711

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_943602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_96582

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
,__inference_block5_conv2_layer_call_fn_96771

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_944122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_94395

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_96562

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_96662

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_96782

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_94238

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
??
?*
 __inference__wrapped_model_94143
input_27
3vgg16_keypoint_detector_tf_nn_bias_add_biasadd_biasc
Ivgg16_keypoint_detector_vgg16_block1_conv1_conv2d_readvariableop_resource:@X
Jvgg16_keypoint_detector_vgg16_block1_conv1_biasadd_readvariableop_resource:@c
Ivgg16_keypoint_detector_vgg16_block1_conv2_conv2d_readvariableop_resource:@@X
Jvgg16_keypoint_detector_vgg16_block1_conv2_biasadd_readvariableop_resource:@d
Ivgg16_keypoint_detector_vgg16_block2_conv1_conv2d_readvariableop_resource:@?Y
Jvgg16_keypoint_detector_vgg16_block2_conv1_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block2_conv2_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block2_conv2_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block3_conv1_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block3_conv1_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block3_conv2_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block3_conv2_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block3_conv3_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block3_conv3_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block4_conv1_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block4_conv1_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block4_conv2_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block4_conv2_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block4_conv3_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block4_conv3_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block5_conv1_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block5_conv1_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block5_conv2_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block5_conv2_biasadd_readvariableop_resource:	?e
Ivgg16_keypoint_detector_vgg16_block5_conv3_conv2d_readvariableop_resource:??Y
Jvgg16_keypoint_detector_vgg16_block5_conv3_biasadd_readvariableop_resource:	?l
Qvgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_resource:?n
Svgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_1_resource:?V
Hvgg16_keypoint_detector_separable_conv2d_biasadd_readvariableop_resource:m
Svgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_resource:o
Uvgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:X
Jvgg16_keypoint_detector_separable_conv2d_1_biasadd_readvariableop_resource:
identity???vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp?Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp?Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1?Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp?Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp?Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1?Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp?Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp?@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp?
Dvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack?
Fvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_1?
Fvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ????2H
Fvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_2?
>vgg16_keypoint_detector/tf.__operators__.getitem/strided_sliceStridedSliceinput_2Mvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack:output:0Ovgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_1:output:0Ovgg16_keypoint_detector/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask2@
>vgg16_keypoint_detector/tf.__operators__.getitem/strided_slice?
.vgg16_keypoint_detector/tf.nn.bias_add/BiasAddBiasAddGvgg16_keypoint_detector/tf.__operators__.getitem/strided_slice:output:03vgg16_keypoint_detector_tf_nn_bias_add_biasadd_bias*
T0*1
_output_shapes
:???????????20
.vgg16_keypoint_detector/tf.nn.bias_add/BiasAdd?
@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02B
@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block1_conv1/Conv2DConv2D7vgg16_keypoint_detector/tf.nn.bias_add/BiasAdd:output:0Hvgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D?
Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block1_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@24
2vgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd?
/vgg16_keypoint_detector/vgg16/block1_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@21
/vgg16_keypoint_detector/vgg16/block1_conv1/Relu?
@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02B
@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block1_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block1_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D?
Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block1_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@24
2vgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd?
/vgg16_keypoint_detector/vgg16/block1_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@21
/vgg16_keypoint_detector/vgg16/block1_conv2/Relu?
1vgg16_keypoint_detector/vgg16/block1_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
23
1vgg16_keypoint_detector/vgg16/block1_pool/MaxPool?
@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block2_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block1_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D?
Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block2_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?24
2vgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd?
/vgg16_keypoint_detector/vgg16/block2_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?21
/vgg16_keypoint_detector/vgg16/block2_conv1/Relu?
@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block2_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block2_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D?
Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block2_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?24
2vgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd?
/vgg16_keypoint_detector/vgg16/block2_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?21
/vgg16_keypoint_detector/vgg16/block2_conv2/Relu?
1vgg16_keypoint_detector/vgg16/block2_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
23
1vgg16_keypoint_detector/vgg16/block2_pool/MaxPool?
@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block3_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block2_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D?
Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block3_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?24
2vgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd?
/vgg16_keypoint_detector/vgg16/block3_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?21
/vgg16_keypoint_detector/vgg16/block3_conv1/Relu?
@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block3_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block3_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D?
Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block3_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?24
2vgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd?
/vgg16_keypoint_detector/vgg16/block3_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?21
/vgg16_keypoint_detector/vgg16/block3_conv2/Relu?
@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block3_conv3/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block3_conv2/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D?
Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block3_conv3/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?24
2vgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd?
/vgg16_keypoint_detector/vgg16/block3_conv3/ReluRelu;vgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?21
/vgg16_keypoint_detector/vgg16/block3_conv3/Relu?
1vgg16_keypoint_detector/vgg16/block3_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
23
1vgg16_keypoint_detector/vgg16/block3_pool/MaxPool?
@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block4_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block3_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D?
Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block4_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????24
2vgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd?
/vgg16_keypoint_detector/vgg16/block4_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????21
/vgg16_keypoint_detector/vgg16/block4_conv1/Relu?
@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block4_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block4_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D?
Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block4_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????24
2vgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd?
/vgg16_keypoint_detector/vgg16/block4_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????21
/vgg16_keypoint_detector/vgg16/block4_conv2/Relu?
@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block4_conv3/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block4_conv2/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D?
Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block4_conv3/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????24
2vgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd?
/vgg16_keypoint_detector/vgg16/block4_conv3/ReluRelu;vgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????21
/vgg16_keypoint_detector/vgg16/block4_conv3/Relu?
1vgg16_keypoint_detector/vgg16/block4_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
23
1vgg16_keypoint_detector/vgg16/block4_pool/MaxPool?
@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block5_conv1/Conv2DConv2D:vgg16_keypoint_detector/vgg16/block4_pool/MaxPool:output:0Hvgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D?
Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block5_conv1/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????24
2vgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd?
/vgg16_keypoint_detector/vgg16/block5_conv1/ReluRelu;vgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????21
/vgg16_keypoint_detector/vgg16/block5_conv1/Relu?
@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block5_conv2/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block5_conv1/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D?
Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block5_conv2/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????24
2vgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd?
/vgg16_keypoint_detector/vgg16/block5_conv2/ReluRelu;vgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????21
/vgg16_keypoint_detector/vgg16/block5_conv2/Relu?
@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOpIvgg16_keypoint_detector_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp?
1vgg16_keypoint_detector/vgg16/block5_conv3/Conv2DConv2D=vgg16_keypoint_detector/vgg16/block5_conv2/Relu:activations:0Hvgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D?
Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/vgg16/block5_conv3/BiasAddBiasAdd:vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D:output:0Ivgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????24
2vgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd?
/vgg16_keypoint_detector/vgg16/block5_conv3/ReluRelu;vgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????21
/vgg16_keypoint_detector/vgg16/block5_conv3/Relu?
1vgg16_keypoint_detector/vgg16/block5_pool/MaxPoolMaxPool=vgg16_keypoint_detector/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
23
1vgg16_keypoint_detector/vgg16/block5_pool/MaxPool?
(vgg16_keypoint_detector/dropout/IdentityIdentity:vgg16_keypoint_detector/vgg16/block5_pool/MaxPool:output:0*
T0*0
_output_shapes
:??????????2*
(vgg16_keypoint_detector/dropout/Identity?
Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpQvgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02J
Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp?
Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpSvgg16_keypoint_detector_separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:?*
dtype02L
Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1?
?vgg16_keypoint_detector/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2A
?vgg16_keypoint_detector/separable_conv2d/separable_conv2d/Shape?
Gvgg16_keypoint_detector/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gvgg16_keypoint_detector/separable_conv2d/separable_conv2d/dilation_rate?
Cvgg16_keypoint_detector/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative1vgg16_keypoint_detector/dropout/Identity:output:0Pvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2E
Cvgg16_keypoint_detector/separable_conv2d/separable_conv2d/depthwise?
9vgg16_keypoint_detector/separable_conv2d/separable_conv2dConv2DLvgg16_keypoint_detector/separable_conv2d/separable_conv2d/depthwise:output:0Rvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2;
9vgg16_keypoint_detector/separable_conv2d/separable_conv2d?
?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOpHvgg16_keypoint_detector_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp?
0vgg16_keypoint_detector/separable_conv2d/BiasAddBiasAddBvgg16_keypoint_detector/separable_conv2d/separable_conv2d:output:0Gvgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22
0vgg16_keypoint_detector/separable_conv2d/BiasAdd?
-vgg16_keypoint_detector/separable_conv2d/ReluRelu9vgg16_keypoint_detector/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2/
-vgg16_keypoint_detector/separable_conv2d/Relu?
Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpSvgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02L
Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp?
Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpUvgg16_keypoint_detector_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02N
Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
Avgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2C
Avgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/Shape?
Ivgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2K
Ivgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/dilation_rate?
Evgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative;vgg16_keypoint_detector/separable_conv2d/Relu:activations:0Rvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2G
Evgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/depthwise?
;vgg16_keypoint_detector/separable_conv2d_1/separable_conv2dConv2DNvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/depthwise:output:0Tvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2=
;vgg16_keypoint_detector/separable_conv2d_1/separable_conv2d?
Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJvgg16_keypoint_detector_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp?
2vgg16_keypoint_detector/separable_conv2d_1/BiasAddBiasAddDvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d:output:0Ivgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????24
2vgg16_keypoint_detector/separable_conv2d_1/BiasAdd?
/vgg16_keypoint_detector/separable_conv2d_1/ReluRelu;vgg16_keypoint_detector/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????21
/vgg16_keypoint_detector/separable_conv2d_1/Relu?
IdentityIdentity=vgg16_keypoint_detector/separable_conv2d_1/Relu:activations:0@^vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOpI^vgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOpK^vgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1B^vgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOpK^vgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOpM^vgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1B^vgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOpB^vgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOpA^vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp?vgg16_keypoint_detector/separable_conv2d/BiasAdd/ReadVariableOp2?
Hvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOpHvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp2?
Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_1Jvgg16_keypoint_detector/separable_conv2d/separable_conv2d/ReadVariableOp_12?
Avgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/separable_conv2d_1/BiasAdd/ReadVariableOp2?
Jvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOpJvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp2?
Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Lvgg16_keypoint_detector/separable_conv2d_1/separable_conv2d/ReadVariableOp_12?
Avgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block1_conv1/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block1_conv1/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block1_conv2/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block1_conv2/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block2_conv1/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block2_conv1/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block2_conv2/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block2_conv2/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block3_conv1/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block3_conv1/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block3_conv2/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block3_conv2/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block3_conv3/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block3_conv3/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block4_conv1/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block4_conv1/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block4_conv2/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block4_conv2/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block4_conv3/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block4_conv3/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block5_conv1/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block5_conv1/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block5_conv2/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block5_conv2/Conv2D/ReadVariableOp2?
Avgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOpAvgg16_keypoint_detector/vgg16/block5_conv3/BiasAdd/ReadVariableOp2?
@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp@vgg16_keypoint_detector/vgg16/block5_conv3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2: 

_output_shapes
:
?
?
%__inference_vgg16_layer_call_fn_96447

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
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
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_944372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?i
?
__inference__traced_save_96968
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

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop7savev2_adam_separable_conv2d_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_1_bias_m_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop7savev2_adam_separable_conv2d_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_1_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?::::: : : : : :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?: : :?:?:::::?:?::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:-)
'
_output_shapes
:?: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::
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
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:. *
(
_output_shapes
:??:!!

_output_shapes	
:?:."*
(
_output_shapes
:??:!#

_output_shapes	
:?:.$*
(
_output_shapes
:??:!%

_output_shapes	
:?:&

_output_shapes
: :'

_output_shapes
: :-()
'
_output_shapes
:?:-))
'
_output_shapes
:?: *

_output_shapes
::,+(
&
_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::-.)
'
_output_shapes
:?:-/)
'
_output_shapes
:?: 0

_output_shapes
::,1(
&
_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::4

_output_shapes
: 
?
?
,__inference_block3_conv2_layer_call_fn_96651

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_943082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_94360

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_96762

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_separable_conv2d_1_layer_call_fn_95073

inputs!
unknown:#
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_950612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_block5_conv3_layer_call_fn_96791

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_944292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_96531

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_952502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_94308

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_94412

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_96702

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
#__inference_signature_wrapper_95779
input_2
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?%

unknown_26:?%

unknown_27:?

unknown_28:$

unknown_29:$

unknown_30:

unknown_31:
identity??StatefulPartitionedCall?
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
:?????????*B
_read_only_resource_inputs$
" 	
 !*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_941432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2: 

_output_shapes
:
?
?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_96742

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block2_conv2_layer_call_fn_96611

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_942732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
%__inference_vgg16_layer_call_fn_94492
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
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
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_944372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_94185

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_94256

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_94377

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_vgg16_layer_call_fn_96504

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
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
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_947552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_block3_conv1_layer_call_fn_96631

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_942912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_94161

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_96642

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_94273

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_94291

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?	
7__inference_vgg16_keypoint_detector_layer_call_fn_96119

inputs
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?%

unknown_26:?%

unknown_27:?

unknown_28:$

unknown_29:$

unknown_30:

unknown_31:
identity??StatefulPartitionedCall?
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
:?????????*B
_read_only_resource_inputs$
" 	
 !*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_951612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_96542

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv1_layer_call_fn_96691

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_943432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95061

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_96602

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?	
7__inference_vgg16_keypoint_detector_layer_call_fn_95230
input_2
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?%

unknown_26:?%

unknown_27:?

unknown_28:$

unknown_29:$

unknown_30:

unknown_31:
identity??StatefulPartitionedCall?
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
:?????????*B
_read_only_resource_inputs$
" 	
 !*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_951612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2: 

_output_shapes
:
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_96521

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_94343

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
7__inference_vgg16_keypoint_detector_layer_call_fn_95544
input_2
unknown#
	unknown_0:@
	unknown_1:@#
	unknown_2:@@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?%

unknown_26:?%

unknown_27:?

unknown_28:$

unknown_29:$

unknown_30:

unknown_31:
identity??StatefulPartitionedCall?
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
:?????????*B
_read_only_resource_inputs$
" 	
 !*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_954042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2: 

_output_shapes
:
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_94173

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?"
!__inference__traced_restore_97131
file_prefixM
2assignvariableop_separable_conv2d_depthwise_kernel:?O
4assignvariableop_1_separable_conv2d_pointwise_kernel:?6
(assignvariableop_2_separable_conv2d_bias:P
6assignvariableop_3_separable_conv2d_1_depthwise_kernel:P
6assignvariableop_4_separable_conv2d_1_pointwise_kernel:8
*assignvariableop_5_separable_conv2d_1_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: A
'assignvariableop_11_block1_conv1_kernel:@3
%assignvariableop_12_block1_conv1_bias:@A
'assignvariableop_13_block1_conv2_kernel:@@3
%assignvariableop_14_block1_conv2_bias:@B
'assignvariableop_15_block2_conv1_kernel:@?4
%assignvariableop_16_block2_conv1_bias:	?C
'assignvariableop_17_block2_conv2_kernel:??4
%assignvariableop_18_block2_conv2_bias:	?C
'assignvariableop_19_block3_conv1_kernel:??4
%assignvariableop_20_block3_conv1_bias:	?C
'assignvariableop_21_block3_conv2_kernel:??4
%assignvariableop_22_block3_conv2_bias:	?C
'assignvariableop_23_block3_conv3_kernel:??4
%assignvariableop_24_block3_conv3_bias:	?C
'assignvariableop_25_block4_conv1_kernel:??4
%assignvariableop_26_block4_conv1_bias:	?C
'assignvariableop_27_block4_conv2_kernel:??4
%assignvariableop_28_block4_conv2_bias:	?C
'assignvariableop_29_block4_conv3_kernel:??4
%assignvariableop_30_block4_conv3_bias:	?C
'assignvariableop_31_block5_conv1_kernel:??4
%assignvariableop_32_block5_conv1_bias:	?C
'assignvariableop_33_block5_conv2_kernel:??4
%assignvariableop_34_block5_conv2_bias:	?C
'assignvariableop_35_block5_conv3_kernel:??4
%assignvariableop_36_block5_conv3_bias:	?#
assignvariableop_37_total: #
assignvariableop_38_count: W
<assignvariableop_39_adam_separable_conv2d_depthwise_kernel_m:?W
<assignvariableop_40_adam_separable_conv2d_pointwise_kernel_m:?>
0assignvariableop_41_adam_separable_conv2d_bias_m:X
>assignvariableop_42_adam_separable_conv2d_1_depthwise_kernel_m:X
>assignvariableop_43_adam_separable_conv2d_1_pointwise_kernel_m:@
2assignvariableop_44_adam_separable_conv2d_1_bias_m:W
<assignvariableop_45_adam_separable_conv2d_depthwise_kernel_v:?W
<assignvariableop_46_adam_separable_conv2d_pointwise_kernel_v:?>
0assignvariableop_47_adam_separable_conv2d_bias_v:X
>assignvariableop_48_adam_separable_conv2d_1_depthwise_kernel_v:X
>assignvariableop_49_adam_separable_conv2d_1_pointwise_kernel_v:@
2assignvariableop_50_adam_separable_conv2d_1_bias_v:
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp2assignvariableop_separable_conv2d_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp4assignvariableop_1_separable_conv2d_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_separable_conv2d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_separable_conv2d_1_depthwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp6assignvariableop_4_separable_conv2d_1_pointwise_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_separable_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_block1_conv1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_block1_conv1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block1_conv2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block1_conv2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block2_conv1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block2_conv1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block2_conv2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block2_conv2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block3_conv1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block3_conv1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block3_conv2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block3_conv2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block3_conv3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block3_conv3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block4_conv1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block4_conv1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block4_conv2_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block4_conv2_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block4_conv3_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block4_conv3_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block5_conv1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block5_conv1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block5_conv2_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_block5_conv2_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block5_conv3_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_block5_conv3_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp<assignvariableop_39_adam_separable_conv2d_depthwise_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp<assignvariableop_40_adam_separable_conv2d_pointwise_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_separable_conv2d_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_separable_conv2d_1_depthwise_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_separable_conv2d_1_pointwise_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_separable_conv2d_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_separable_conv2d_depthwise_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp<assignvariableop_46_adam_separable_conv2d_pointwise_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp0assignvariableop_47_adam_separable_conv2d_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_separable_conv2d_1_depthwise_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_separable_conv2d_1_pointwise_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_separable_conv2d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51?	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
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
_user_specified_namefile_prefix
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_95250

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block4_pool_layer_call_fn_94191

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_941852
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_94429

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block1_conv1_layer_call_fn_96551

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_942212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_2:
serving_default_input_2:0???????????N
separable_conv2d_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
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

regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"name": "vgg16_keypoint_detector", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "vgg16_keypoint_detector", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem", "inbound_nodes": [["input_2", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"class_name": "__ellipsis__"}, {"start": null, "stop": null, "step": -1}]}}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.bias_add", "trainable": true, "dtype": "float32", "function": "nn.bias_add"}, "name": "tf.nn.bias_add", "inbound_nodes": [["tf.__operators__.getitem", 0, 0, {"bias": [-103.93900299072266, -116.77899932861328, -123.68000030517578], "data_format": "NHWC"}]]}, {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "name": "vgg16", "inbound_nodes": [[["tf.nn.bias_add", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["vgg16", 1, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_1", "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["separable_conv2d_1", 0, 0]]}, "shared_object_id": 60, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vgg16_keypoint_detector", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem", "inbound_nodes": [["input_2", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"class_name": "__ellipsis__"}, {"start": null, "stop": null, "step": -1}]}}]], "shared_object_id": 1}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.bias_add", "trainable": true, "dtype": "float32", "function": "nn.bias_add"}, "name": "tf.nn.bias_add", "inbound_nodes": [["tf.__operators__.getitem", 0, 0, {"bias": [-103.93900299072266, -116.77899932861328, -123.68000030517578], "data_format": "NHWC"}]], "shared_object_id": 2}, {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "name": "vgg16", "inbound_nodes": [[["tf.nn.bias_add", 0, 0, {}]]], "shared_object_id": 48}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["vgg16", 1, 0, {}]]], "shared_object_id": 49}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_1", "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]], "shared_object_id": 59}], "input_layers": [["input_2", 0, 0]], "output_layers": [["separable_conv2d_1", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.__operators__.getitem", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "inbound_nodes": [["input_2", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"class_name": "__ellipsis__"}, {"start": null, "stop": null, "step": -1}]}}]], "shared_object_id": 1}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.nn.bias_add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.nn.bias_add", "trainable": true, "dtype": "float32", "function": "nn.bias_add"}, "inbound_nodes": [["tf.__operators__.getitem", 0, 0, {"bias": [-103.93900299072266, -116.77899932861328, -123.68000030517578], "data_format": "NHWC"}]], "shared_object_id": 2}
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
 layer_with_weights-11
 layer-16
!layer_with_weights-12
!layer-17
"layer-18
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"name": "vgg16", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "inbound_nodes": [[["tf.nn.bias_add", 0, 0, {}]]], "shared_object_id": 48, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]], "shared_object_id": 37}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]], "shared_object_id": 46}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]], "shared_object_id": 47}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}}}
?
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["vgg16", 1, 0, {}]]], "shared_object_id": 49}
?
+depthwise_kernel
,pointwise_kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "separable_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 512]}}
?
2depthwise_kernel
3pointwise_kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "separable_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]], "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 8]}}
?
9iter

:beta_1

;beta_2
	<decay
=learning_rate+m?,m?-m?2m?3m?4m?+v?,v?-v?2v?3v?4v?"
	optimizer
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
+26
,27
-28
229
330
431"
trackable_list_wrapper
 "
trackable_list_wrapper
J
+0
,1
-2
23
34
45"
trackable_list_wrapper
?
		variables

regularization_losses
Xnon_trainable_variables
trainable_variables
Ymetrics
Zlayer_regularization_losses
[layer_metrics

\layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

>kernel
?bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}}
?

@kernel
Abias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_conv1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 64]}}
?
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "block1_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block1_conv2", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 67}}
?

Bkernel
Cbias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_pool", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 64]}}
?

Dkernel
Ebias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_conv1", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 128]}}
?
q	variables
rregularization_losses
strainable_variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block2_conv2", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 70}}
?

Fkernel
Gbias
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_pool", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 128]}}
?

Hkernel
Ibias
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block3_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv1", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 256]}}
?

Jkernel
Kbias
}	variables
~regularization_losses
trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block3_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv2", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 256]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "block3_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block3_conv3", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 74}}
?

Lkernel
Mbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block4_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_pool", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 256]}}
?

Nkernel
Obias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block4_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv1", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 512]}}
?

Pkernel
Qbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block4_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv2", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 512]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "block4_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block4_conv3", 0, 0, {}]]], "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 78}}
?

Rkernel
Sbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block5_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_pool", 0, 0, {}]]], "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 79}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?

Tkernel
Ubias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block5_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv1", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?

Vkernel
Wbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "block5_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv2", 0, 0, {}]]], "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "block5_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block5_conv3", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 82}}
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#	variables
$regularization_losses
?non_trainable_variables
%trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'	variables
(regularization_losses
?non_trainable_variables
)trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<::?2!separable_conv2d/depthwise_kernel
<::?2!separable_conv2d/pointwise_kernel
#:!2separable_conv2d/bias
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
.	variables
/regularization_losses
?non_trainable_variables
0trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
=:;2#separable_conv2d_1/depthwise_kernel
=:;2#separable_conv2d_1/pointwise_kernel
%:#2separable_conv2d_1/bias
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
?
5	variables
6regularization_losses
?non_trainable_variables
7trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
/:-??2block5_conv2/kernel
 :?2block5_conv2/bias
/:-??2block5_conv3/kernel
 :?2block5_conv3/bias
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]	variables
^regularization_losses
?non_trainable_variables
_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
a	variables
bregularization_losses
?non_trainable_variables
ctrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
e	variables
fregularization_losses
?non_trainable_variables
gtrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
i	variables
jregularization_losses
?non_trainable_variables
ktrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
m	variables
nregularization_losses
?non_trainable_variables
otrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
q	variables
rregularization_losses
?non_trainable_variables
strainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
u	variables
vregularization_losses
?non_trainable_variables
wtrainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
y	variables
zregularization_losses
?non_trainable_variables
{trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
}	variables
~regularization_losses
?non_trainable_variables
trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18"
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
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 83}
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
@0
A1"
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
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
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
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
J0
K1"
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
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
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
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
V0
W1"
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
A:??2(Adam/separable_conv2d/depthwise_kernel/m
A:??2(Adam/separable_conv2d/pointwise_kernel/m
(:&2Adam/separable_conv2d/bias/m
B:@2*Adam/separable_conv2d_1/depthwise_kernel/m
B:@2*Adam/separable_conv2d_1/pointwise_kernel/m
*:(2Adam/separable_conv2d_1/bias/m
A:??2(Adam/separable_conv2d/depthwise_kernel/v
A:??2(Adam/separable_conv2d/pointwise_kernel/v
(:&2Adam/separable_conv2d/bias/v
B:@2*Adam/separable_conv2d_1/depthwise_kernel/v
B:@2*Adam/separable_conv2d_1/pointwise_kernel/v
*:(2Adam/separable_conv2d_1/bias/v
?2?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95910
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_96048
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95622
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95700?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_vgg16_keypoint_detector_layer_call_fn_95230
7__inference_vgg16_keypoint_detector_layer_call_fn_96119
7__inference_vgg16_keypoint_detector_layer_call_fn_96190
7__inference_vgg16_keypoint_detector_layer_call_fn_95544?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_94143?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_2???????????
?2?
@__inference_vgg16_layer_call_and_return_conditional_losses_96290
@__inference_vgg16_layer_call_and_return_conditional_losses_96390
@__inference_vgg16_layer_call_and_return_conditional_losses_94941
@__inference_vgg16_layer_call_and_return_conditional_losses_95015?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_vgg16_layer_call_fn_94492
%__inference_vgg16_layer_call_fn_96447
%__inference_vgg16_layer_call_fn_96504
%__inference_vgg16_layer_call_fn_94867?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_96509
B__inference_dropout_layer_call_and_return_conditional_losses_96521?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_96526
'__inference_dropout_layer_call_fn_96531?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
0__inference_separable_conv2d_layer_call_fn_95044?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95061?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
2__inference_separable_conv2d_1_layer_call_fn_95073?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?B?
#__inference_signature_wrapper_95779input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_96542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block1_conv1_layer_call_fn_96551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_96562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block1_conv2_layer_call_fn_96571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_pool_layer_call_and_return_conditional_losses_94149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block1_pool_layer_call_fn_94155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_96582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block2_conv1_layer_call_fn_96591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_96602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block2_conv2_layer_call_fn_96611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_pool_layer_call_and_return_conditional_losses_94161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block2_pool_layer_call_fn_94167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_96622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv1_layer_call_fn_96631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_96642?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv2_layer_call_fn_96651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_96662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv3_layer_call_fn_96671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_pool_layer_call_and_return_conditional_losses_94173?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block3_pool_layer_call_fn_94179?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_96682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv1_layer_call_fn_96691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_96702?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv2_layer_call_fn_96711?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_96722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv3_layer_call_fn_96731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_pool_layer_call_and_return_conditional_losses_94185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block4_pool_layer_call_fn_94191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_96742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv1_layer_call_fn_96751?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_96762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv2_layer_call_fn_96771?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_96782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv3_layer_call_fn_96791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_pool_layer_call_and_return_conditional_losses_94197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block5_pool_layer_call_fn_94203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
	J
Const?
 __inference__wrapped_model_94143?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234:?7
0?-
+?(
input_2???????????
? "O?L
J
separable_conv2d_14?1
separable_conv2d_1??????????
G__inference_block1_conv1_layer_call_and_return_conditional_losses_96542p>?9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
,__inference_block1_conv1_layer_call_fn_96551c>?9?6
/?,
*?'
inputs???????????
? ""????????????@?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_96562p@A9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
,__inference_block1_conv2_layer_call_fn_96571c@A9?6
/?,
*?'
inputs???????????@
? ""????????????@?
F__inference_block1_pool_layer_call_and_return_conditional_losses_94149?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block1_pool_layer_call_fn_94155?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block2_conv1_layer_call_and_return_conditional_losses_96582mBC7?4
-?*
(?%
inputs?????????pp@
? ".?+
$?!
0?????????pp?
? ?
,__inference_block2_conv1_layer_call_fn_96591`BC7?4
-?*
(?%
inputs?????????pp@
? "!??????????pp??
G__inference_block2_conv2_layer_call_and_return_conditional_losses_96602nDE8?5
.?+
)?&
inputs?????????pp?
? ".?+
$?!
0?????????pp?
? ?
,__inference_block2_conv2_layer_call_fn_96611aDE8?5
.?+
)?&
inputs?????????pp?
? "!??????????pp??
F__inference_block2_pool_layer_call_and_return_conditional_losses_94161?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block2_pool_layer_call_fn_94167?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block3_conv1_layer_call_and_return_conditional_losses_96622nFG8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
,__inference_block3_conv1_layer_call_fn_96631aFG8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
G__inference_block3_conv2_layer_call_and_return_conditional_losses_96642nHI8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
,__inference_block3_conv2_layer_call_fn_96651aHI8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
G__inference_block3_conv3_layer_call_and_return_conditional_losses_96662nJK8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
,__inference_block3_conv3_layer_call_fn_96671aJK8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
F__inference_block3_pool_layer_call_and_return_conditional_losses_94173?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block3_pool_layer_call_fn_94179?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block4_conv1_layer_call_and_return_conditional_losses_96682nLM8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv1_layer_call_fn_96691aLM8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv2_layer_call_and_return_conditional_losses_96702nNO8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv2_layer_call_fn_96711aNO8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv3_layer_call_and_return_conditional_losses_96722nPQ8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv3_layer_call_fn_96731aPQ8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_pool_layer_call_and_return_conditional_losses_94185?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block4_pool_layer_call_fn_94191?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block5_conv1_layer_call_and_return_conditional_losses_96742nRS8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv1_layer_call_fn_96751aRS8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv2_layer_call_and_return_conditional_losses_96762nTU8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv2_layer_call_fn_96771aTU8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv3_layer_call_and_return_conditional_losses_96782nVW8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv3_layer_call_fn_96791aVW8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block5_pool_layer_call_and_return_conditional_losses_94197?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block5_pool_layer_call_fn_94203?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_dropout_layer_call_and_return_conditional_losses_96509n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_96521n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
'__inference_dropout_layer_call_fn_96526a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
'__inference_dropout_layer_call_fn_96531a<?9
2?/
)?&
inputs??????????
p
? "!????????????
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95061?234I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_separable_conv2d_1_layer_call_fn_95073?234I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95032?+,-J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
0__inference_separable_conv2d_layer_call_fn_95044?+,-J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
#__inference_signature_wrapper_95779?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234E?B
? 
;?8
6
input_2+?(
input_2???????????"O?L
J
separable_conv2d_14?1
separable_conv2d_1??????????
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95622?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234B??
8?5
+?(
input_2???????????
p 

 
? "-?*
#? 
0?????????
? ?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95700?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234B??
8?5
+?(
input_2???????????
p

 
? "-?*
#? 
0?????????
? ?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_95910?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234A?>
7?4
*?'
inputs???????????
p 

 
? "-?*
#? 
0?????????
? ?
R__inference_vgg16_keypoint_detector_layer_call_and_return_conditional_losses_96048?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234A?>
7?4
*?'
inputs???????????
p

 
? "-?*
#? 
0?????????
? ?
7__inference_vgg16_keypoint_detector_layer_call_fn_95230?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234B??
8?5
+?(
input_2???????????
p 

 
? " ???????????
7__inference_vgg16_keypoint_detector_layer_call_fn_95544?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234B??
8?5
+?(
input_2???????????
p

 
? " ???????????
7__inference_vgg16_keypoint_detector_layer_call_fn_96119?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234A?>
7?4
*?'
inputs???????????
p 

 
? " ???????????
7__inference_vgg16_keypoint_detector_layer_call_fn_96190?"?>?@ABCDEFGHIJKLMNOPQRSTUVW+,-234A?>
7?4
*?'
inputs???????????
p

 
? " ???????????
@__inference_vgg16_layer_call_and_return_conditional_losses_94941?>?@ABCDEFGHIJKLMNOPQRSTUVWB??
8?5
+?(
input_1???????????
p 

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg16_layer_call_and_return_conditional_losses_95015?>?@ABCDEFGHIJKLMNOPQRSTUVWB??
8?5
+?(
input_1???????????
p

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg16_layer_call_and_return_conditional_losses_96290?>?@ABCDEFGHIJKLMNOPQRSTUVWA?>
7?4
*?'
inputs???????????
p 

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg16_layer_call_and_return_conditional_losses_96390?>?@ABCDEFGHIJKLMNOPQRSTUVWA?>
7?4
*?'
inputs???????????
p

 
? ".?+
$?!
0??????????
? ?
%__inference_vgg16_layer_call_fn_94492?>?@ABCDEFGHIJKLMNOPQRSTUVWB??
8?5
+?(
input_1???????????
p 

 
? "!????????????
%__inference_vgg16_layer_call_fn_94867?>?@ABCDEFGHIJKLMNOPQRSTUVWB??
8?5
+?(
input_1???????????
p

 
? "!????????????
%__inference_vgg16_layer_call_fn_96447?>?@ABCDEFGHIJKLMNOPQRSTUVWA?>
7?4
*?'
inputs???????????
p 

 
? "!????????????
%__inference_vgg16_layer_call_fn_96504?>?@ABCDEFGHIJKLMNOPQRSTUVWA?>
7?4
*?'
inputs???????????
p

 
? "!???????????
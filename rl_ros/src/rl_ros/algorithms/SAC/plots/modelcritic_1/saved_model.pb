
í
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
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
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8ª

critic_network/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	"*.
shared_namecritic_network/dense_4/kernel

1critic_network/dense_4/kernel/Read/ReadVariableOpReadVariableOpcritic_network/dense_4/kernel*
_output_shapes
:	"*
dtype0

critic_network/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecritic_network/dense_4/bias

/critic_network/dense_4/bias/Read/ReadVariableOpReadVariableOpcritic_network/dense_4/bias*
_output_shapes	
:*
dtype0

critic_network/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_namecritic_network/dense_5/kernel

1critic_network/dense_5/kernel/Read/ReadVariableOpReadVariableOpcritic_network/dense_5/kernel* 
_output_shapes
:
*
dtype0

critic_network/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecritic_network/dense_5/bias

/critic_network/dense_5/bias/Read/ReadVariableOpReadVariableOpcritic_network/dense_5/bias*
_output_shapes	
:*
dtype0

critic_network/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_namecritic_network/dense_6/kernel

1critic_network/dense_6/kernel/Read/ReadVariableOpReadVariableOpcritic_network/dense_6/kernel*
_output_shapes
:	*
dtype0

critic_network/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecritic_network/dense_6/bias

/critic_network/dense_6/bias/Read/ReadVariableOpReadVariableOpcritic_network/dense_6/bias*
_output_shapes
:*
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
¥
$Adam/critic_network/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	"*5
shared_name&$Adam/critic_network/dense_4/kernel/m

8Adam/critic_network/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/critic_network/dense_4/kernel/m*
_output_shapes
:	"*
dtype0

"Adam/critic_network/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/critic_network/dense_4/bias/m

6Adam/critic_network/dense_4/bias/m/Read/ReadVariableOpReadVariableOp"Adam/critic_network/dense_4/bias/m*
_output_shapes	
:*
dtype0
¦
$Adam/critic_network/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/critic_network/dense_5/kernel/m

8Adam/critic_network/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/critic_network/dense_5/kernel/m* 
_output_shapes
:
*
dtype0

"Adam/critic_network/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/critic_network/dense_5/bias/m

6Adam/critic_network/dense_5/bias/m/Read/ReadVariableOpReadVariableOp"Adam/critic_network/dense_5/bias/m*
_output_shapes	
:*
dtype0
¥
$Adam/critic_network/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$Adam/critic_network/dense_6/kernel/m

8Adam/critic_network/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/critic_network/dense_6/kernel/m*
_output_shapes
:	*
dtype0

"Adam/critic_network/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/critic_network/dense_6/bias/m

6Adam/critic_network/dense_6/bias/m/Read/ReadVariableOpReadVariableOp"Adam/critic_network/dense_6/bias/m*
_output_shapes
:*
dtype0
¥
$Adam/critic_network/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	"*5
shared_name&$Adam/critic_network/dense_4/kernel/v

8Adam/critic_network/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/critic_network/dense_4/kernel/v*
_output_shapes
:	"*
dtype0

"Adam/critic_network/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/critic_network/dense_4/bias/v

6Adam/critic_network/dense_4/bias/v/Read/ReadVariableOpReadVariableOp"Adam/critic_network/dense_4/bias/v*
_output_shapes	
:*
dtype0
¦
$Adam/critic_network/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/critic_network/dense_5/kernel/v

8Adam/critic_network/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/critic_network/dense_5/kernel/v* 
_output_shapes
:
*
dtype0

"Adam/critic_network/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/critic_network/dense_5/bias/v

6Adam/critic_network/dense_5/bias/v/Read/ReadVariableOpReadVariableOp"Adam/critic_network/dense_5/bias/v*
_output_shapes	
:*
dtype0
¥
$Adam/critic_network/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$Adam/critic_network/dense_6/kernel/v

8Adam/critic_network/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/critic_network/dense_6/kernel/v*
_output_shapes
:	*
dtype0

"Adam/critic_network/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/critic_network/dense_6/bias/v

6Adam/critic_network/dense_6/bias/v/Read/ReadVariableOpReadVariableOp"Adam/critic_network/dense_6/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ô
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¯
value¥B¢ B

fc1
fc2
q
	optimizer
loss
trainable_variables
regularization_losses
	variables
		keras_api


signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
¬
iter

beta_1

beta_2
	 decay
!learning_ratem6m7m8m9m:m;v<v=v>v?v@vA
 
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
­
"layer_regularization_losses
#layer_metrics

$layers
trainable_variables
%non_trainable_variables
regularization_losses
	variables
&metrics
 
XV
VARIABLE_VALUEcritic_network/dense_4/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcritic_network/dense_4/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
'layer_regularization_losses
(layer_metrics

)layers
trainable_variables
*non_trainable_variables
regularization_losses
	variables
+metrics
XV
VARIABLE_VALUEcritic_network/dense_5/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcritic_network/dense_5/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
,layer_regularization_losses
-layer_metrics

.layers
trainable_variables
/non_trainable_variables
regularization_losses
	variables
0metrics
VT
VARIABLE_VALUEcritic_network/dense_6/kernel#q/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcritic_network/dense_6/bias!q/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
1layer_regularization_losses
2layer_metrics

3layers
trainable_variables
4non_trainable_variables
regularization_losses
	variables
5metrics
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
 
 

0
1
2
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
 
 
{y
VARIABLE_VALUE$Adam/critic_network/dense_4/kernel/mAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE"Adam/critic_network/dense_4/bias/m?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$Adam/critic_network/dense_5/kernel/mAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE"Adam/critic_network/dense_5/bias/m?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE$Adam/critic_network/dense_6/kernel/m?q/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE"Adam/critic_network/dense_6/bias/m=q/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$Adam/critic_network/dense_4/kernel/vAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE"Adam/critic_network/dense_4/bias/v?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$Adam/critic_network/dense_5/kernel/vAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE"Adam/critic_network/dense_5/bias/v?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE$Adam/critic_network/dense_6/kernel/v?q/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE"Adam/critic_network/dense_6/bias/v=q/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ 
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2critic_network/dense_4/kernelcritic_network/dense_4/biascritic_network/dense_5/kernelcritic_network/dense_5/biascritic_network/dense_6/kernelcritic_network/dense_6/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_136213
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¸
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1critic_network/dense_4/kernel/Read/ReadVariableOp/critic_network/dense_4/bias/Read/ReadVariableOp1critic_network/dense_5/kernel/Read/ReadVariableOp/critic_network/dense_5/bias/Read/ReadVariableOp1critic_network/dense_6/kernel/Read/ReadVariableOp/critic_network/dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp8Adam/critic_network/dense_4/kernel/m/Read/ReadVariableOp6Adam/critic_network/dense_4/bias/m/Read/ReadVariableOp8Adam/critic_network/dense_5/kernel/m/Read/ReadVariableOp6Adam/critic_network/dense_5/bias/m/Read/ReadVariableOp8Adam/critic_network/dense_6/kernel/m/Read/ReadVariableOp6Adam/critic_network/dense_6/bias/m/Read/ReadVariableOp8Adam/critic_network/dense_4/kernel/v/Read/ReadVariableOp6Adam/critic_network/dense_4/bias/v/Read/ReadVariableOp8Adam/critic_network/dense_5/kernel/v/Read/ReadVariableOp6Adam/critic_network/dense_5/bias/v/Read/ReadVariableOp8Adam/critic_network/dense_6/kernel/v/Read/ReadVariableOp6Adam/critic_network/dense_6/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__traced_save_136365
ç
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_network/dense_4/kernelcritic_network/dense_4/biascritic_network/dense_5/kernelcritic_network/dense_5/biascritic_network/dense_6/kernelcritic_network/dense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate$Adam/critic_network/dense_4/kernel/m"Adam/critic_network/dense_4/bias/m$Adam/critic_network/dense_5/kernel/m"Adam/critic_network/dense_5/bias/m$Adam/critic_network/dense_6/kernel/m"Adam/critic_network/dense_6/bias/m$Adam/critic_network/dense_4/kernel/v"Adam/critic_network/dense_4/bias/v$Adam/critic_network/dense_5/kernel/v"Adam/critic_network/dense_5/bias/v$Adam/critic_network/dense_6/kernel/v"Adam/critic_network/dense_6/bias/v*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_136444ñ»
	
Ü
C__inference_dense_6_layer_call_and_return_conditional_losses_136263

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò9
ê
__inference__traced_save_136365
file_prefix<
8savev2_critic_network_dense_4_kernel_read_readvariableop:
6savev2_critic_network_dense_4_bias_read_readvariableop<
8savev2_critic_network_dense_5_kernel_read_readvariableop:
6savev2_critic_network_dense_5_bias_read_readvariableop<
8savev2_critic_network_dense_6_kernel_read_readvariableop:
6savev2_critic_network_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopC
?savev2_adam_critic_network_dense_4_kernel_m_read_readvariableopA
=savev2_adam_critic_network_dense_4_bias_m_read_readvariableopC
?savev2_adam_critic_network_dense_5_kernel_m_read_readvariableopA
=savev2_adam_critic_network_dense_5_bias_m_read_readvariableopC
?savev2_adam_critic_network_dense_6_kernel_m_read_readvariableopA
=savev2_adam_critic_network_dense_6_bias_m_read_readvariableopC
?savev2_adam_critic_network_dense_4_kernel_v_read_readvariableopA
=savev2_adam_critic_network_dense_4_bias_v_read_readvariableopC
?savev2_adam_critic_network_dense_5_kernel_v_read_readvariableopA
=savev2_adam_critic_network_dense_5_bias_v_read_readvariableopC
?savev2_adam_critic_network_dense_6_kernel_v_read_readvariableopA
=savev2_adam_critic_network_dense_6_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameö

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

valueþ	Bû	B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB#q/kernel/.ATTRIBUTES/VARIABLE_VALUEB!q/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¸
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesó
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_critic_network_dense_4_kernel_read_readvariableop6savev2_critic_network_dense_4_bias_read_readvariableop8savev2_critic_network_dense_5_kernel_read_readvariableop6savev2_critic_network_dense_5_bias_read_readvariableop8savev2_critic_network_dense_6_kernel_read_readvariableop6savev2_critic_network_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop?savev2_adam_critic_network_dense_4_kernel_m_read_readvariableop=savev2_adam_critic_network_dense_4_bias_m_read_readvariableop?savev2_adam_critic_network_dense_5_kernel_m_read_readvariableop=savev2_adam_critic_network_dense_5_bias_m_read_readvariableop?savev2_adam_critic_network_dense_6_kernel_m_read_readvariableop=savev2_adam_critic_network_dense_6_bias_m_read_readvariableop?savev2_adam_critic_network_dense_4_kernel_v_read_readvariableop=savev2_adam_critic_network_dense_4_bias_v_read_readvariableop?savev2_adam_critic_network_dense_5_kernel_v_read_readvariableop=savev2_adam_critic_network_dense_5_bias_v_read_readvariableop?savev2_adam_critic_network_dense_6_kernel_v_read_readvariableop=savev2_adam_critic_network_dense_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Å
_input_shapes³
°: :	"::
::	:: : : : : :	"::
::	::	"::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	":!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::
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
: :%!

_output_shapes
:	":!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	":!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
¹
Ã
$__inference_signature_wrapper_136213
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_1360782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
á
}
(__inference_dense_6_layer_call_fn_136272

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1361492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
Ü
C__inference_dense_4_layer_call_and_return_conditional_losses_136224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	"*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ"::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
e

"__inference__traced_restore_136444
file_prefix2
.assignvariableop_critic_network_dense_4_kernel2
.assignvariableop_1_critic_network_dense_4_bias4
0assignvariableop_2_critic_network_dense_5_kernel2
.assignvariableop_3_critic_network_dense_5_bias4
0assignvariableop_4_critic_network_dense_6_kernel2
.assignvariableop_5_critic_network_dense_6_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate<
8assignvariableop_11_adam_critic_network_dense_4_kernel_m:
6assignvariableop_12_adam_critic_network_dense_4_bias_m<
8assignvariableop_13_adam_critic_network_dense_5_kernel_m:
6assignvariableop_14_adam_critic_network_dense_5_bias_m<
8assignvariableop_15_adam_critic_network_dense_6_kernel_m:
6assignvariableop_16_adam_critic_network_dense_6_bias_m<
8assignvariableop_17_adam_critic_network_dense_4_kernel_v:
6assignvariableop_18_adam_critic_network_dense_4_bias_v<
8assignvariableop_19_adam_critic_network_dense_5_kernel_v:
6assignvariableop_20_adam_critic_network_dense_5_bias_v<
8assignvariableop_21_adam_critic_network_dense_6_kernel_v:
6assignvariableop_22_adam_critic_network_dense_6_bias_v
identity_24¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ü

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

valueþ	Bû	B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB#q/kernel/.ATTRIBUTES/VARIABLE_VALUEB!q/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¾
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices£
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity­
AssignVariableOpAssignVariableOp.assignvariableop_critic_network_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1³
AssignVariableOp_1AssignVariableOp.assignvariableop_1_critic_network_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2µ
AssignVariableOp_2AssignVariableOp0assignvariableop_2_critic_network_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_critic_network_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4µ
AssignVariableOp_4AssignVariableOp0assignvariableop_4_critic_network_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_critic_network_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¡
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11À
AssignVariableOp_11AssignVariableOp8assignvariableop_11_adam_critic_network_dense_4_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¾
AssignVariableOp_12AssignVariableOp6assignvariableop_12_adam_critic_network_dense_4_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13À
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_critic_network_dense_5_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¾
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_critic_network_dense_5_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15À
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adam_critic_network_dense_6_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¾
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_critic_network_dense_6_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17À
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_critic_network_dense_4_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¾
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_critic_network_dense_4_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19À
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_critic_network_dense_5_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¾
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_critic_network_dense_5_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21À
AssignVariableOp_21AssignVariableOp8assignvariableop_21_adam_critic_network_dense_6_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¾
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_critic_network_dense_6_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpØ
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23Ë
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ã
}
(__inference_dense_5_layer_call_fn_136253

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1361232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
}
(__inference_dense_4_layer_call_fn_136233

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1360962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ"::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
ß(
Ð
!__inference__wrapped_model_136078
input_1
input_29
5critic_network_dense_4_matmul_readvariableop_resource:
6critic_network_dense_4_biasadd_readvariableop_resource9
5critic_network_dense_5_matmul_readvariableop_resource:
6critic_network_dense_5_biasadd_readvariableop_resource9
5critic_network_dense_6_matmul_readvariableop_resource:
6critic_network_dense_6_biasadd_readvariableop_resource
identity¢-critic_network/dense_4/BiasAdd/ReadVariableOp¢,critic_network/dense_4/MatMul/ReadVariableOp¢-critic_network/dense_5/BiasAdd/ReadVariableOp¢,critic_network/dense_5/MatMul/ReadVariableOp¢-critic_network/dense_6/BiasAdd/ReadVariableOp¢,critic_network/dense_6/MatMul/ReadVariableOpz
critic_network/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
critic_network/concat/axis¬
critic_network/concatConcatV2input_1input_2#critic_network/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
critic_network/concatÓ
,critic_network/dense_4/MatMul/ReadVariableOpReadVariableOp5critic_network_dense_4_matmul_readvariableop_resource*
_output_shapes
:	"*
dtype02.
,critic_network/dense_4/MatMul/ReadVariableOpÑ
critic_network/dense_4/MatMulMatMulcritic_network/concat:output:04critic_network/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
critic_network/dense_4/MatMulÒ
-critic_network/dense_4/BiasAdd/ReadVariableOpReadVariableOp6critic_network_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-critic_network/dense_4/BiasAdd/ReadVariableOpÞ
critic_network/dense_4/BiasAddBiasAdd'critic_network/dense_4/MatMul:product:05critic_network/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
critic_network/dense_4/BiasAdd
critic_network/dense_4/ReluRelu'critic_network/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
critic_network/dense_4/ReluÔ
,critic_network/dense_5/MatMul/ReadVariableOpReadVariableOp5critic_network_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,critic_network/dense_5/MatMul/ReadVariableOpÜ
critic_network/dense_5/MatMulMatMul)critic_network/dense_4/Relu:activations:04critic_network/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
critic_network/dense_5/MatMulÒ
-critic_network/dense_5/BiasAdd/ReadVariableOpReadVariableOp6critic_network_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-critic_network/dense_5/BiasAdd/ReadVariableOpÞ
critic_network/dense_5/BiasAddBiasAdd'critic_network/dense_5/MatMul:product:05critic_network/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
critic_network/dense_5/BiasAdd
critic_network/dense_5/ReluRelu'critic_network/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
critic_network/dense_5/ReluÓ
,critic_network/dense_6/MatMul/ReadVariableOpReadVariableOp5critic_network_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,critic_network/dense_6/MatMul/ReadVariableOpÛ
critic_network/dense_6/MatMulMatMul)critic_network/dense_5/Relu:activations:04critic_network/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
critic_network/dense_6/MatMulÑ
-critic_network/dense_6/BiasAdd/ReadVariableOpReadVariableOp6critic_network_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-critic_network/dense_6/BiasAdd/ReadVariableOpÝ
critic_network/dense_6/BiasAddBiasAdd'critic_network/dense_6/MatMul:product:05critic_network/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
critic_network/dense_6/BiasAdd
IdentityIdentity'critic_network/dense_6/BiasAdd:output:0.^critic_network/dense_4/BiasAdd/ReadVariableOp-^critic_network/dense_4/MatMul/ReadVariableOp.^critic_network/dense_5/BiasAdd/ReadVariableOp-^critic_network/dense_5/MatMul/ReadVariableOp.^critic_network/dense_6/BiasAdd/ReadVariableOp-^critic_network/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::2^
-critic_network/dense_4/BiasAdd/ReadVariableOp-critic_network/dense_4/BiasAdd/ReadVariableOp2\
,critic_network/dense_4/MatMul/ReadVariableOp,critic_network/dense_4/MatMul/ReadVariableOp2^
-critic_network/dense_5/BiasAdd/ReadVariableOp-critic_network/dense_5/BiasAdd/ReadVariableOp2\
,critic_network/dense_5/MatMul/ReadVariableOp,critic_network/dense_5/MatMul/ReadVariableOp2^
-critic_network/dense_6/BiasAdd/ReadVariableOp-critic_network/dense_6/BiasAdd/ReadVariableOp2\
,critic_network/dense_6/MatMul/ReadVariableOp,critic_network/dense_6/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ö	
Ü
C__inference_dense_5_layer_call_and_return_conditional_losses_136244

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Õ
J__inference_critic_network_layer_call_and_return_conditional_losses_136166
input_1
input_2
dense_4_136107
dense_4_136109
dense_5_136134
dense_5_136136
dense_6_136160
dense_6_136162
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2input_1input_2concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
concat
dense_4/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_4_136107dense_4_136109*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1360962!
dense_4/StatefulPartitionedCall·
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_136134dense_5_136136*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1361232!
dense_5/StatefulPartitionedCall¶
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_136160dense_6_136162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1361492!
dense_6/StatefulPartitionedCallâ
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
	
Ü
C__inference_dense_6_layer_call_and_return_conditional_losses_136149

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
Ü
C__inference_dense_5_layer_call_and_return_conditional_losses_136123

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
Î
/__inference_critic_network_layer_call_fn_136185
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_critic_network_layer_call_and_return_conditional_losses_1361662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ó	
Ü
C__inference_dense_4_layer_call_and_return_conditional_losses_136096

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	"*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ"::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*è
serving_defaultÔ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ 
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¯]
º
fc1
fc2
q
	optimizer
loss
trainable_variables
regularization_losses
	variables
		keras_api


signatures
B_default_save_signature
*C&call_and_return_all_conditional_losses
D__call__"Ì
_tf_keras_model²{"class_name": "CriticNetwork", "name": "critic_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CriticNetwork"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003000000142492354, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ï

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*E&call_and_return_all_conditional_losses
F__call__"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 34]}}
ñ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 512]}}
ñ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 512]}}
¿
iter

beta_1

beta_2
	 decay
!learning_ratem6m7m8m9m:m;v<v=v>v?v@vA"
	optimizer
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
Ê
"layer_regularization_losses
#layer_metrics

$layers
trainable_variables
%non_trainable_variables
regularization_losses
	variables
&metrics
D__call__
B_default_save_signature
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
,
Kserving_default"
signature_map
0:.	"2critic_network/dense_4/kernel
*:(2critic_network/dense_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
'layer_regularization_losses
(layer_metrics

)layers
trainable_variables
*non_trainable_variables
regularization_losses
	variables
+metrics
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
1:/
2critic_network/dense_5/kernel
*:(2critic_network/dense_5/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
,layer_regularization_losses
-layer_metrics

.layers
trainable_variables
/non_trainable_variables
regularization_losses
	variables
0metrics
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
0:.	2critic_network/dense_6/kernel
):'2critic_network/dense_6/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
1layer_regularization_losses
2layer_metrics

3layers
trainable_variables
4non_trainable_variables
regularization_losses
	variables
5metrics
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5:3	"2$Adam/critic_network/dense_4/kernel/m
/:-2"Adam/critic_network/dense_4/bias/m
6:4
2$Adam/critic_network/dense_5/kernel/m
/:-2"Adam/critic_network/dense_5/bias/m
5:3	2$Adam/critic_network/dense_6/kernel/m
.:,2"Adam/critic_network/dense_6/bias/m
5:3	"2$Adam/critic_network/dense_4/kernel/v
/:-2"Adam/critic_network/dense_4/bias/v
6:4
2$Adam/critic_network/dense_5/kernel/v
/:-2"Adam/critic_network/dense_5/bias/v
5:3	2$Adam/critic_network/dense_6/kernel/v
.:,2"Adam/critic_network/dense_6/bias/v
2
!__inference__wrapped_model_136078Þ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *N¢K
I¢F
!
input_1ÿÿÿÿÿÿÿÿÿ 
!
input_2ÿÿÿÿÿÿÿÿÿ
À2½
J__inference_critic_network_layer_call_and_return_conditional_losses_136166î
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
annotationsª *N¢K
I¢F
!
input_1ÿÿÿÿÿÿÿÿÿ 
!
input_2ÿÿÿÿÿÿÿÿÿ
¥2¢
/__inference_critic_network_layer_call_fn_136185î
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
annotationsª *N¢K
I¢F
!
input_1ÿÿÿÿÿÿÿÿÿ 
!
input_2ÿÿÿÿÿÿÿÿÿ
í2ê
C__inference_dense_4_layer_call_and_return_conditional_losses_136224¢
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
Ò2Ï
(__inference_dense_4_layer_call_fn_136233¢
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
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_136244¢
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
Ò2Ï
(__inference_dense_5_layer_call_fn_136253¢
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
í2ê
C__inference_dense_6_layer_call_and_return_conditional_losses_136263¢
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
Ò2Ï
(__inference_dense_6_layer_call_fn_136272¢
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
ÒBÏ
$__inference_signature_wrapper_136213input_1input_2"
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
 ½
!__inference__wrapped_model_136078X¢U
N¢K
I¢F
!
input_1ÿÿÿÿÿÿÿÿÿ 
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿØ
J__inference_critic_network_layer_call_and_return_conditional_losses_136166X¢U
N¢K
I¢F
!
input_1ÿÿÿÿÿÿÿÿÿ 
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¯
/__inference_critic_network_layer_call_fn_136185|X¢U
N¢K
I¢F
!
input_1ÿÿÿÿÿÿÿÿÿ 
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_4_layer_call_and_return_conditional_losses_136224]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ"
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_4_layer_call_fn_136233P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ"
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_5_layer_call_and_return_conditional_losses_136244^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_5_layer_call_fn_136253Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_6_layer_call_and_return_conditional_losses_136263]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_6_layer_call_fn_136272P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÑ
$__inference_signature_wrapper_136213¨i¢f
¢ 
_ª\
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ 
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
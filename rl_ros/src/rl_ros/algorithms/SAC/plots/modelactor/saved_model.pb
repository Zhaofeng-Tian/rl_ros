ĝ
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8��
�
actor_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*+
shared_nameactor_network/dense/kernel
�
.actor_network/dense/kernel/Read/ReadVariableOpReadVariableOpactor_network/dense/kernel*
_output_shapes
:	 �*
dtype0
�
actor_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameactor_network/dense/bias
�
,actor_network/dense/bias/Read/ReadVariableOpReadVariableOpactor_network/dense/bias*
_output_shapes	
:�*
dtype0
�
actor_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameactor_network/dense_1/kernel
�
0actor_network/dense_1/kernel/Read/ReadVariableOpReadVariableOpactor_network/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
actor_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameactor_network/dense_1/bias
�
.actor_network/dense_1/bias/Read/ReadVariableOpReadVariableOpactor_network/dense_1/bias*
_output_shapes	
:�*
dtype0
�
actor_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameactor_network/dense_2/kernel
�
0actor_network/dense_2/kernel/Read/ReadVariableOpReadVariableOpactor_network/dense_2/kernel*
_output_shapes
:	�*
dtype0
�
actor_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameactor_network/dense_2/bias
�
.actor_network/dense_2/bias/Read/ReadVariableOpReadVariableOpactor_network/dense_2/bias*
_output_shapes
:*
dtype0
�
actor_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameactor_network/dense_3/kernel
�
0actor_network/dense_3/kernel/Read/ReadVariableOpReadVariableOpactor_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
actor_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameactor_network/dense_3/bias
�
.actor_network/dense_3/bias/Read/ReadVariableOpReadVariableOpactor_network/dense_3/bias*
_output_shapes
:*
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
�
!Adam/actor_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*2
shared_name#!Adam/actor_network/dense/kernel/m
�
5Adam/actor_network/dense/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense/kernel/m*
_output_shapes
:	 �*
dtype0
�
Adam/actor_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/actor_network/dense/bias/m
�
3Adam/actor_network/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/actor_network/dense/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/actor_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/actor_network/dense_1/kernel/m
�
7Adam/actor_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/actor_network/dense_1/kernel/m* 
_output_shapes
:
��*
dtype0
�
!Adam/actor_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/actor_network/dense_1/bias/m
�
5Adam/actor_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/actor_network/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/actor_network/dense_2/kernel/m
�
7Adam/actor_network/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/actor_network/dense_2/kernel/m*
_output_shapes
:	�*
dtype0
�
!Adam/actor_network/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/actor_network/dense_2/bias/m
�
5Adam/actor_network/dense_2/bias/m/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense_2/bias/m*
_output_shapes
:*
dtype0
�
#Adam/actor_network/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/actor_network/dense_3/kernel/m
�
7Adam/actor_network/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/actor_network/dense_3/kernel/m*
_output_shapes
:	�*
dtype0
�
!Adam/actor_network/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/actor_network/dense_3/bias/m
�
5Adam/actor_network/dense_3/bias/m/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense_3/bias/m*
_output_shapes
:*
dtype0
�
!Adam/actor_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*2
shared_name#!Adam/actor_network/dense/kernel/v
�
5Adam/actor_network/dense/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense/kernel/v*
_output_shapes
:	 �*
dtype0
�
Adam/actor_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/actor_network/dense/bias/v
�
3Adam/actor_network/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/actor_network/dense/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/actor_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/actor_network/dense_1/kernel/v
�
7Adam/actor_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/actor_network/dense_1/kernel/v* 
_output_shapes
:
��*
dtype0
�
!Adam/actor_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/actor_network/dense_1/bias/v
�
5Adam/actor_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense_1/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/actor_network/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/actor_network/dense_2/kernel/v
�
7Adam/actor_network/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/actor_network/dense_2/kernel/v*
_output_shapes
:	�*
dtype0
�
!Adam/actor_network/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/actor_network/dense_2/bias/v
�
5Adam/actor_network/dense_2/bias/v/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense_2/bias/v*
_output_shapes
:*
dtype0
�
#Adam/actor_network/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/actor_network/dense_3/kernel/v
�
7Adam/actor_network/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/actor_network/dense_3/kernel/v*
_output_shapes
:	�*
dtype0
�
!Adam/actor_network/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/actor_network/dense_3/bias/v
�
5Adam/actor_network/dense_3/bias/v/Read/ReadVariableOpReadVariableOp!Adam/actor_network/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�'
value�'B�' B�'
�
fc1
fc2
mu
	sigma
	optimizer
loss
trainable_variables
regularization_losses
		variables

	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemBmCmDmEmFmGmHmIvJvKvLvMvNvOvPvQ
 
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
�
)layer_regularization_losses
*layer_metrics

+layers
trainable_variables
,non_trainable_variables
regularization_losses
		variables
-metrics
 
US
VARIABLE_VALUEactor_network/dense/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEactor_network/dense/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
.layer_regularization_losses
/layer_metrics

0layers
trainable_variables
1non_trainable_variables
regularization_losses
	variables
2metrics
WU
VARIABLE_VALUEactor_network/dense_1/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEactor_network/dense_1/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
3layer_regularization_losses
4layer_metrics

5layers
trainable_variables
6non_trainable_variables
regularization_losses
	variables
7metrics
VT
VARIABLE_VALUEactor_network/dense_2/kernel$mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEactor_network/dense_2/bias"mu/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
8layer_regularization_losses
9layer_metrics

:layers
trainable_variables
;non_trainable_variables
regularization_losses
	variables
<metrics
YW
VARIABLE_VALUEactor_network/dense_3/kernel'sigma/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEactor_network/dense_3/bias%sigma/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
=layer_regularization_losses
>layer_metrics

?layers
 trainable_variables
@non_trainable_variables
!regularization_losses
"	variables
Ametrics
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

0
1
2
3
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
 
 
 
 
 
xv
VARIABLE_VALUE!Adam/actor_network/dense/kernel/mAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/actor_network/dense/bias/m?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/actor_network/dense_1/kernel/mAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE!Adam/actor_network/dense_1/bias/m?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#Adam/actor_network/dense_2/kernel/m@mu/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE!Adam/actor_network/dense_2/bias/m>mu/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/actor_network/dense_3/kernel/mCsigma/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/actor_network/dense_3/bias/mAsigma/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/actor_network/dense/kernel/vAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/actor_network/dense/bias/v?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/actor_network/dense_1/kernel/vAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE!Adam/actor_network/dense_1/bias/v?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#Adam/actor_network/dense_2/kernel/v@mu/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE!Adam/actor_network/dense_2/bias/v>mu/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/actor_network/dense_3/kernel/vCsigma/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/actor_network/dense_3/bias/vAsigma/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:��������� *
dtype0*
shape:��������� 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1actor_network/dense/kernelactor_network/dense/biasactor_network/dense_1/kernelactor_network/dense_1/biasactor_network/dense_2/kernelactor_network/dense_2/biasactor_network/dense_3/kernelactor_network/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_135625
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.actor_network/dense/kernel/Read/ReadVariableOp,actor_network/dense/bias/Read/ReadVariableOp0actor_network/dense_1/kernel/Read/ReadVariableOp.actor_network/dense_1/bias/Read/ReadVariableOp0actor_network/dense_2/kernel/Read/ReadVariableOp.actor_network/dense_2/bias/Read/ReadVariableOp0actor_network/dense_3/kernel/Read/ReadVariableOp.actor_network/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp5Adam/actor_network/dense/kernel/m/Read/ReadVariableOp3Adam/actor_network/dense/bias/m/Read/ReadVariableOp7Adam/actor_network/dense_1/kernel/m/Read/ReadVariableOp5Adam/actor_network/dense_1/bias/m/Read/ReadVariableOp7Adam/actor_network/dense_2/kernel/m/Read/ReadVariableOp5Adam/actor_network/dense_2/bias/m/Read/ReadVariableOp7Adam/actor_network/dense_3/kernel/m/Read/ReadVariableOp5Adam/actor_network/dense_3/bias/m/Read/ReadVariableOp5Adam/actor_network/dense/kernel/v/Read/ReadVariableOp3Adam/actor_network/dense/bias/v/Read/ReadVariableOp7Adam/actor_network/dense_1/kernel/v/Read/ReadVariableOp5Adam/actor_network/dense_1/bias/v/Read/ReadVariableOp7Adam/actor_network/dense_2/kernel/v/Read/ReadVariableOp5Adam/actor_network/dense_2/bias/v/Read/ReadVariableOp7Adam/actor_network/dense_3/kernel/v/Read/ReadVariableOp5Adam/actor_network/dense_3/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
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
GPU2 *0J 8� *(
f#R!
__inference__traced_save_135814
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactor_network/dense/kernelactor_network/dense/biasactor_network/dense_1/kernelactor_network/dense_1/biasactor_network/dense_2/kernelactor_network/dense_2/biasactor_network/dense_3/kernelactor_network/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate!Adam/actor_network/dense/kernel/mAdam/actor_network/dense/bias/m#Adam/actor_network/dense_1/kernel/m!Adam/actor_network/dense_1/bias/m#Adam/actor_network/dense_2/kernel/m!Adam/actor_network/dense_2/bias/m#Adam/actor_network/dense_3/kernel/m!Adam/actor_network/dense_3/bias/m!Adam/actor_network/dense/kernel/vAdam/actor_network/dense/bias/v#Adam/actor_network/dense_1/kernel/v!Adam/actor_network/dense_1/bias/v#Adam/actor_network/dense_2/kernel/v!Adam/actor_network/dense_2/bias/v#Adam/actor_network/dense_3/kernel/v!Adam/actor_network/dense_3/bias/v*)
Tin"
 2*
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
GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_135911��
�
}
(__inference_dense_1_layer_call_fn_135665

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1354942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_135636

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
}
(__inference_dense_2_layer_call_fn_135684

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1355202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�8
�
!__inference__wrapped_model_135452
input_16
2actor_network_dense_matmul_readvariableop_resource7
3actor_network_dense_biasadd_readvariableop_resource8
4actor_network_dense_1_matmul_readvariableop_resource9
5actor_network_dense_1_biasadd_readvariableop_resource8
4actor_network_dense_2_matmul_readvariableop_resource9
5actor_network_dense_2_biasadd_readvariableop_resource8
4actor_network_dense_3_matmul_readvariableop_resource9
5actor_network_dense_3_biasadd_readvariableop_resource
identity

identity_1��*actor_network/dense/BiasAdd/ReadVariableOp�)actor_network/dense/MatMul/ReadVariableOp�,actor_network/dense_1/BiasAdd/ReadVariableOp�+actor_network/dense_1/MatMul/ReadVariableOp�,actor_network/dense_2/BiasAdd/ReadVariableOp�+actor_network/dense_2/MatMul/ReadVariableOp�,actor_network/dense_3/BiasAdd/ReadVariableOp�+actor_network/dense_3/MatMul/ReadVariableOp�
)actor_network/dense/MatMul/ReadVariableOpReadVariableOp2actor_network_dense_matmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02+
)actor_network/dense/MatMul/ReadVariableOp�
actor_network/dense/MatMulMatMulinput_11actor_network/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor_network/dense/MatMul�
*actor_network/dense/BiasAdd/ReadVariableOpReadVariableOp3actor_network_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*actor_network/dense/BiasAdd/ReadVariableOp�
actor_network/dense/BiasAddBiasAdd$actor_network/dense/MatMul:product:02actor_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor_network/dense/BiasAdd�
actor_network/dense/ReluRelu$actor_network/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actor_network/dense/Relu�
+actor_network/dense_1/MatMul/ReadVariableOpReadVariableOp4actor_network_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+actor_network/dense_1/MatMul/ReadVariableOp�
actor_network/dense_1/MatMulMatMul&actor_network/dense/Relu:activations:03actor_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor_network/dense_1/MatMul�
,actor_network/dense_1/BiasAdd/ReadVariableOpReadVariableOp5actor_network_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,actor_network/dense_1/BiasAdd/ReadVariableOp�
actor_network/dense_1/BiasAddBiasAdd&actor_network/dense_1/MatMul:product:04actor_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor_network/dense_1/BiasAdd�
actor_network/dense_1/ReluRelu&actor_network/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actor_network/dense_1/Relu�
+actor_network/dense_2/MatMul/ReadVariableOpReadVariableOp4actor_network_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+actor_network/dense_2/MatMul/ReadVariableOp�
actor_network/dense_2/MatMulMatMul(actor_network/dense_1/Relu:activations:03actor_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actor_network/dense_2/MatMul�
,actor_network/dense_2/BiasAdd/ReadVariableOpReadVariableOp5actor_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,actor_network/dense_2/BiasAdd/ReadVariableOp�
actor_network/dense_2/BiasAddBiasAdd&actor_network/dense_2/MatMul:product:04actor_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actor_network/dense_2/BiasAdd�
+actor_network/dense_3/MatMul/ReadVariableOpReadVariableOp4actor_network_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+actor_network/dense_3/MatMul/ReadVariableOp�
actor_network/dense_3/MatMulMatMul(actor_network/dense_1/Relu:activations:03actor_network/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actor_network/dense_3/MatMul�
,actor_network/dense_3/BiasAdd/ReadVariableOpReadVariableOp5actor_network_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,actor_network/dense_3/BiasAdd/ReadVariableOp�
actor_network/dense_3/BiasAddBiasAdd&actor_network/dense_3/MatMul:product:04actor_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actor_network/dense_3/BiasAdd�
%actor_network/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%actor_network/clip_by_value/Minimum/y�
#actor_network/clip_by_value/MinimumMinimum&actor_network/dense_3/BiasAdd:output:0.actor_network/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������2%
#actor_network/clip_by_value/Minimum�
actor_network/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
actor_network/clip_by_value/y�
actor_network/clip_by_valueMaximum'actor_network/clip_by_value/Minimum:z:0&actor_network/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������2
actor_network/clip_by_value�
IdentityIdentity&actor_network/dense_2/BiasAdd:output:0+^actor_network/dense/BiasAdd/ReadVariableOp*^actor_network/dense/MatMul/ReadVariableOp-^actor_network/dense_1/BiasAdd/ReadVariableOp,^actor_network/dense_1/MatMul/ReadVariableOp-^actor_network/dense_2/BiasAdd/ReadVariableOp,^actor_network/dense_2/MatMul/ReadVariableOp-^actor_network/dense_3/BiasAdd/ReadVariableOp,^actor_network/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identityactor_network/clip_by_value:z:0+^actor_network/dense/BiasAdd/ReadVariableOp*^actor_network/dense/MatMul/ReadVariableOp-^actor_network/dense_1/BiasAdd/ReadVariableOp,^actor_network/dense_1/MatMul/ReadVariableOp-^actor_network/dense_2/BiasAdd/ReadVariableOp,^actor_network/dense_2/MatMul/ReadVariableOp-^actor_network/dense_3/BiasAdd/ReadVariableOp,^actor_network/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:��������� ::::::::2X
*actor_network/dense/BiasAdd/ReadVariableOp*actor_network/dense/BiasAdd/ReadVariableOp2V
)actor_network/dense/MatMul/ReadVariableOp)actor_network/dense/MatMul/ReadVariableOp2\
,actor_network/dense_1/BiasAdd/ReadVariableOp,actor_network/dense_1/BiasAdd/ReadVariableOp2Z
+actor_network/dense_1/MatMul/ReadVariableOp+actor_network/dense_1/MatMul/ReadVariableOp2\
,actor_network/dense_2/BiasAdd/ReadVariableOp,actor_network/dense_2/BiasAdd/ReadVariableOp2Z
+actor_network/dense_2/MatMul/ReadVariableOp+actor_network/dense_2/MatMul/ReadVariableOp2\
,actor_network/dense_3/BiasAdd/ReadVariableOp,actor_network/dense_3/BiasAdd/ReadVariableOp2Z
+actor_network/dense_3/MatMul/ReadVariableOp+actor_network/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_1
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_135494

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
.__inference_actor_network_layer_call_fn_135592
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_actor_network_layer_call_and_return_conditional_losses_1355682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_1
�~
�
"__inference__traced_restore_135911
file_prefix/
+assignvariableop_actor_network_dense_kernel/
+assignvariableop_1_actor_network_dense_bias3
/assignvariableop_2_actor_network_dense_1_kernel1
-assignvariableop_3_actor_network_dense_1_bias3
/assignvariableop_4_actor_network_dense_2_kernel1
-assignvariableop_5_actor_network_dense_2_bias3
/assignvariableop_6_actor_network_dense_3_kernel1
-assignvariableop_7_actor_network_dense_3_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate9
5assignvariableop_13_adam_actor_network_dense_kernel_m7
3assignvariableop_14_adam_actor_network_dense_bias_m;
7assignvariableop_15_adam_actor_network_dense_1_kernel_m9
5assignvariableop_16_adam_actor_network_dense_1_bias_m;
7assignvariableop_17_adam_actor_network_dense_2_kernel_m9
5assignvariableop_18_adam_actor_network_dense_2_bias_m;
7assignvariableop_19_adam_actor_network_dense_3_kernel_m9
5assignvariableop_20_adam_actor_network_dense_3_bias_m9
5assignvariableop_21_adam_actor_network_dense_kernel_v7
3assignvariableop_22_adam_actor_network_dense_bias_v;
7assignvariableop_23_adam_actor_network_dense_1_kernel_v9
5assignvariableop_24_adam_actor_network_dense_1_bias_v;
7assignvariableop_25_adam_actor_network_dense_2_kernel_v9
5assignvariableop_26_adam_actor_network_dense_2_bias_v;
7assignvariableop_27_adam_actor_network_dense_3_kernel_v9
5assignvariableop_28_adam_actor_network_dense_3_bias_v
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB$mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB"mu/bias/.ATTRIBUTES/VARIABLE_VALUEB'sigma/kernel/.ATTRIBUTES/VARIABLE_VALUEB%sigma/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCsigma/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAsigma/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCsigma/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAsigma/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp+assignvariableop_actor_network_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_actor_network_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_actor_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_actor_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_actor_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_actor_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp/assignvariableop_6_actor_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_actor_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp5assignvariableop_13_adam_actor_network_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp3assignvariableop_14_adam_actor_network_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adam_actor_network_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_actor_network_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_actor_network_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adam_actor_network_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_actor_network_dense_3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_actor_network_dense_3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_actor_network_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_actor_network_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_actor_network_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_actor_network_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_actor_network_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_actor_network_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_actor_network_dense_3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_actor_network_dense_3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29�
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*�
_input_shapesx
v: :::::::::::::::::::::::::::::2$
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
AssignVariableOp_28AssignVariableOp_282(
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
�	
�
C__inference_dense_3_layer_call_and_return_conditional_losses_135694

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_actor_network_layer_call_and_return_conditional_losses_135568
input_1
dense_135478
dense_135480
dense_1_135505
dense_1_135507
dense_2_135531
dense_2_135533
dense_3_135557
dense_3_135559
identity

identity_1��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_135478dense_135480*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1354672
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_135505dense_1_135507*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1354942!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_135531dense_2_135533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1355202!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_135557dense_3_135559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1355462!
dense_3/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum(dense_3/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������2
clip_by_value�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identityclip_by_value:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:��������� ::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_1
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_135656

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_135520

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_135467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
{
&__inference_dense_layer_call_fn_135645

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1354672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�D
�
__inference__traced_save_135814
file_prefix9
5savev2_actor_network_dense_kernel_read_readvariableop7
3savev2_actor_network_dense_bias_read_readvariableop;
7savev2_actor_network_dense_1_kernel_read_readvariableop9
5savev2_actor_network_dense_1_bias_read_readvariableop;
7savev2_actor_network_dense_2_kernel_read_readvariableop9
5savev2_actor_network_dense_2_bias_read_readvariableop;
7savev2_actor_network_dense_3_kernel_read_readvariableop9
5savev2_actor_network_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop@
<savev2_adam_actor_network_dense_kernel_m_read_readvariableop>
:savev2_adam_actor_network_dense_bias_m_read_readvariableopB
>savev2_adam_actor_network_dense_1_kernel_m_read_readvariableop@
<savev2_adam_actor_network_dense_1_bias_m_read_readvariableopB
>savev2_adam_actor_network_dense_2_kernel_m_read_readvariableop@
<savev2_adam_actor_network_dense_2_bias_m_read_readvariableopB
>savev2_adam_actor_network_dense_3_kernel_m_read_readvariableop@
<savev2_adam_actor_network_dense_3_bias_m_read_readvariableop@
<savev2_adam_actor_network_dense_kernel_v_read_readvariableop>
:savev2_adam_actor_network_dense_bias_v_read_readvariableopB
>savev2_adam_actor_network_dense_1_kernel_v_read_readvariableop@
<savev2_adam_actor_network_dense_1_bias_v_read_readvariableopB
>savev2_adam_actor_network_dense_2_kernel_v_read_readvariableop@
<savev2_adam_actor_network_dense_2_bias_v_read_readvariableopB
>savev2_adam_actor_network_dense_3_kernel_v_read_readvariableop@
<savev2_adam_actor_network_dense_3_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB$mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB"mu/bias/.ATTRIBUTES/VARIABLE_VALUEB'sigma/kernel/.ATTRIBUTES/VARIABLE_VALUEB%sigma/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCsigma/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAsigma/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCsigma/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAsigma/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_actor_network_dense_kernel_read_readvariableop3savev2_actor_network_dense_bias_read_readvariableop7savev2_actor_network_dense_1_kernel_read_readvariableop5savev2_actor_network_dense_1_bias_read_readvariableop7savev2_actor_network_dense_2_kernel_read_readvariableop5savev2_actor_network_dense_2_bias_read_readvariableop7savev2_actor_network_dense_3_kernel_read_readvariableop5savev2_actor_network_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop<savev2_adam_actor_network_dense_kernel_m_read_readvariableop:savev2_adam_actor_network_dense_bias_m_read_readvariableop>savev2_adam_actor_network_dense_1_kernel_m_read_readvariableop<savev2_adam_actor_network_dense_1_bias_m_read_readvariableop>savev2_adam_actor_network_dense_2_kernel_m_read_readvariableop<savev2_adam_actor_network_dense_2_bias_m_read_readvariableop>savev2_adam_actor_network_dense_3_kernel_m_read_readvariableop<savev2_adam_actor_network_dense_3_bias_m_read_readvariableop<savev2_adam_actor_network_dense_kernel_v_read_readvariableop:savev2_adam_actor_network_dense_bias_v_read_readvariableop>savev2_adam_actor_network_dense_1_kernel_v_read_readvariableop<savev2_adam_actor_network_dense_1_bias_v_read_readvariableop>savev2_adam_actor_network_dense_2_kernel_v_read_readvariableop<savev2_adam_actor_network_dense_2_bias_v_read_readvariableop>savev2_adam_actor_network_dense_3_kernel_v_read_readvariableop<savev2_adam_actor_network_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	 �:�:
��:�:	�::	�:: : : : : :	 �:�:
��:�:	�::	�::	 �:�:
��:�:	�::	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	 �:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 �:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	 �:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_135675

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_dense_3_layer_call_fn_135703

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1355462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_dense_3_layer_call_and_return_conditional_losses_135546

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
$__inference_signature_wrapper_135625
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_1354522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0��������� <
output_10
StatefulPartitionedCall:0���������<
output_20
StatefulPartitionedCall:1���������tensorflow/serving/predict:�q
�
fc1
fc2
mu
	sigma
	optimizer
loss
trainable_variables
regularization_losses
		variables

	keras_api

signatures
R_default_save_signature
*S&call_and_return_all_conditional_losses
T__call__"�
_tf_keras_model�{"class_name": "ActorNetwork", "name": "actor_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ActorNetwork"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003000000142492354, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*W&call_and_return_all_conditional_losses
X__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
�

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
*[&call_and_return_all_conditional_losses
\__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemBmCmDmEmFmGmHmIvJvKvLvMvNvOvPvQ"
	optimizer
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
)layer_regularization_losses
*layer_metrics

+layers
trainable_variables
,non_trainable_variables
regularization_losses
		variables
-metrics
T__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
]serving_default"
signature_map
-:+	 �2actor_network/dense/kernel
':%�2actor_network/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
.layer_regularization_losses
/layer_metrics

0layers
trainable_variables
1non_trainable_variables
regularization_losses
	variables
2metrics
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
0:.
��2actor_network/dense_1/kernel
):'�2actor_network/dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
3layer_regularization_losses
4layer_metrics

5layers
trainable_variables
6non_trainable_variables
regularization_losses
	variables
7metrics
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
/:-	�2actor_network/dense_2/kernel
(:&2actor_network/dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
8layer_regularization_losses
9layer_metrics

:layers
trainable_variables
;non_trainable_variables
regularization_losses
	variables
<metrics
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
/:-	�2actor_network/dense_3/kernel
(:&2actor_network/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
=layer_regularization_losses
>layer_metrics

?layers
 trainable_variables
@non_trainable_variables
!regularization_losses
"	variables
Ametrics
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
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
<
0
1
2
3"
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
2:0	 �2!Adam/actor_network/dense/kernel/m
,:*�2Adam/actor_network/dense/bias/m
5:3
��2#Adam/actor_network/dense_1/kernel/m
.:,�2!Adam/actor_network/dense_1/bias/m
4:2	�2#Adam/actor_network/dense_2/kernel/m
-:+2!Adam/actor_network/dense_2/bias/m
4:2	�2#Adam/actor_network/dense_3/kernel/m
-:+2!Adam/actor_network/dense_3/bias/m
2:0	 �2!Adam/actor_network/dense/kernel/v
,:*�2Adam/actor_network/dense/bias/v
5:3
��2#Adam/actor_network/dense_1/kernel/v
.:,�2!Adam/actor_network/dense_1/bias/v
4:2	�2#Adam/actor_network/dense_2/kernel/v
-:+2!Adam/actor_network/dense_2/bias/v
4:2	�2#Adam/actor_network/dense_3/kernel/v
-:+2!Adam/actor_network/dense_3/bias/v
�2�
!__inference__wrapped_model_135452�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1��������� 
�2�
I__inference_actor_network_layer_call_and_return_conditional_losses_135568�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1��������� 
�2�
.__inference_actor_network_layer_call_fn_135592�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1��������� 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_135636�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_135645�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_135656�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_135665�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_135675�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_135684�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_3_layer_call_and_return_conditional_losses_135694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_3_layer_call_fn_135703�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_135625input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_135452�0�-
&�#
!�
input_1��������� 
� "c�`
.
output_1"�
output_1���������
.
output_2"�
output_2����������
I__inference_actor_network_layer_call_and_return_conditional_losses_135568�0�-
&�#
!�
input_1��������� 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
.__inference_actor_network_layer_call_fn_135592{0�-
&�#
!�
input_1��������� 
� "=�:
�
0���������
�
1����������
C__inference_dense_1_layer_call_and_return_conditional_losses_135656^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_1_layer_call_fn_135665Q0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_2_layer_call_and_return_conditional_losses_135675]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_dense_2_layer_call_fn_135684P0�-
&�#
!�
inputs����������
� "�����������
C__inference_dense_3_layer_call_and_return_conditional_losses_135694]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_dense_3_layer_call_fn_135703P0�-
&�#
!�
inputs����������
� "�����������
A__inference_dense_layer_call_and_return_conditional_losses_135636]/�,
%�"
 �
inputs��������� 
� "&�#
�
0����������
� z
&__inference_dense_layer_call_fn_135645P/�,
%�"
 �
inputs��������� 
� "������������
$__inference_signature_wrapper_135625�;�8
� 
1�.
,
input_1!�
input_1��������� "c�`
.
output_1"�
output_1���������
.
output_2"�
output_2���������
# 1. model.named_parameters(), iterative printing model.named_parameters() will print the name and param of each iteration element

model = optim = cfg = nn = torch = None

for name, param in model.named_parameters():
	print(name,param.requires_grad)
	param.requires_grad=False

# 2. model.parameters(), iterative printing model.parameters() will print the parameter of each iteration element 
# without printing the name. This is the difference between it and named_parameters, both of which can be used to 
# change the attributes of requirements_grad

for  param in model.parameters():
	print(param.requires_grad)
	param.requires_grad=False


# 3. If model.state_dict().items() prints this option in each iteration, all name and param will be printed, 
# but all param here is requires_grad=False, there is no way to change the attributes of requirements_grad, 
# so change requirements_grad Attributes can only pass the above two methods.

for name, param in model.state_dict().items():
	print(name, param.requires_grad)
    # param.requires_grad = True

# 4. Modify the attributes of optimizer after changing requirements_grad

optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), #Only update the parameters with requirements_grad=True
    lr=cfg.TRAIN.LR,
    momentum=cfg.TRAIN.MOMENTUM,
    weight_decay=cfg.TRAIN.WD,
    nesterov=cfg.TRAIN.NESTEROV
)
        
# 5. Random parameter initialization

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
model.apply(init_weights)
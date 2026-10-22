#carve split
#define hyperparams
#write training function:
#>loop over batches
#>>build train_step:
#>>>write mask
#>>>use part 2's forward pass, but with mask input
#>>>update kernels
#>>>update w_1_2
#>>>return k_update
#>>print no-dropout val acc every 10
#>score once on final weights
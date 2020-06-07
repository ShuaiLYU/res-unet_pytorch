
import  torch

def iter_on_a_batch(batch, model, losses, optim, phase, device):
    assert isinstance(losses, dict)
    assert phase in ["train", "valid", "test", "infer"]
    img_batch, label_pixel_batch, label_batch, file_name_batch = batch
    img_rensor = torch.tensor(img_batch).float().to(device)
    label_tensor = torch.tensor(label_batch).float().to(device)
    # forward
    active=torch.nn.Sigmoid()
    mask_tensor = active(model(img_rensor) ) # 在devie
    mask_batch = mask_tensor.detach().cpu().numpy()  # cpu

    ###### cul loss
    if phase in ["train", "valid", "test"]:


        label_pixel_tensor = torch.tensor(label_pixel_batch).float().to(device)  # gpu


        loss_segment = losses["supervise"](mask_tensor.squeeze(1), label_pixel_tensor.squeeze(1))
        loss_dict = {"segment": loss_segment.mean()}
    ##### backward
    if phase in ["train"]:
        assert isinstance(loss_dict, dict)
        model.zero_grad()
        if len(loss_dict)==1:
            loss_sum=list(loss_dict.values())[0]
        else:
            loss_sum = sum(*list(loss_dict.values()))
        loss_sum.backward()
        optim.step()
        #### return
    result = {"mask_batch": mask_batch, }
    if phase in ["train", "valid", "test"]:
        for key, loss in loss_dict.items():
            loss_dict[key] = float(loss)
    result["loss"] = loss_dict
    return result
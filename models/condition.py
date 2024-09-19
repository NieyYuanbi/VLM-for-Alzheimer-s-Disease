
def conditional_prompt(mri_pred_level,pet_pred_level) -> str:
    if mri_pred_level == 0:
        context_map = {
            0: "The MRI image shows normal; The PET image shows normal",
            1: "The MRI image shows normal; The PET image shows abnormal",
        }
    elif mri_pred_level == 1:
        context_map = {
            0: "The MRI image shows abnormal; The PET image shows normal",
            1: "The MRI image shows abnormal; The PET image shows abnormal",
        }
    else:
        raise "The mri_pred_level not exist!"

    return context_map[pet_pred_level]
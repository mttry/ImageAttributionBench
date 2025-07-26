v1 = "Describe the content of the image in as much detail as possible, focusing on objects, setting, colors, actions, and mood. Keep the description between 20 and 70 words."

# imagenet
v2 = "Using the given label “[LABEL]” as guidance, describe the image in as much detail as possible, focusing on objects, setting, colors, actions, and mood. Incorporate elements related to the label. Keep the description between 20 and 50 words."
def build_labeled_prompt(label:str) -> str:
    return f"Using the given label '{label}' as guidance, describe the image in as much detail as possible, focusing on objects, setting, colors, actions, and mood. Incorporate elements related to the label. Keep the description between 20 and 50 words."

#facial
v3 = (
    "Describe the face in the image with focus on facial features. "
    "Include details such as the shape and appearance of the eyes, nose, mouth, and ears, "
    "the color and texture of the skin or fur, facial expression, and any visible hair or fur. "
    "The description should clearly center on the face, with minimal reference to the background or body. "
    "Keep it vivid and concise, 20 to 50 words."
)

#scene
def build_scene_prompt(scene_label: str) -> str:
    return (
        f"Describe a {scene_label} scene in the image. "
        f"Focus on the main elements typical of a {scene_label}, such as objects, layout, colors, and lighting. "
        f"Avoid focusing on people or irrelevant details. The description should be visual and vivid, 20 to 50 words."
    )

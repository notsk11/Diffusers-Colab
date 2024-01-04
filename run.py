images = pipeline(
    prompt = prompt,
    negative_prompt = negative_prompt,
    guidance_scale = guidance_scale,
    num_inference_steps = num_inference_steps,
    width = selected_resolution[0] - selected_resolution[0] % 8,
    height = selected_resolution[1] - selected_resolution[1] % 8,
    num_images_per_prompt = num_images_per_prompt,
    generator = torch.Generator("cuda").manual_seed(seed),
    ).images

media.show_images(images)
images[0].save("output.jpg")

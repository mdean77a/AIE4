import chainlit as cl


@cl.on_chat_start
async def start():
    image1 = cl.Image(path="fig3.jpg", name="image1", ElementSize= "large", display="inline")
    image2 = cl.Image(path="fig3.jpg", name="image1", ElementSize= "large", display="side")
    image3 = cl.Image(path="fig3.jpg", name="image1", ElementSize= "large", display="page")
    # Attach the image to the message
    await cl.Message(
        content="This message has an image!",
        elements=[image1, image2, image3],
    ).send()

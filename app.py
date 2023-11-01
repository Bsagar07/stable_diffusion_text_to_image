# import required modules
import tkinter as tk
import customtkinter as ctk 
from PIL import ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# user authentication token
token = "Your Auth Token"


# create instance
app = tk.Tk()

app.title("Stable Diffusion Text to Image Application")
app.geometry("532x632")
ctk.set_appearance_mode("dark")


prompt = ctk.CTkEntry(master=app, width=512, height=40, corner_radius=10, fg_color="white")
prompt.place(x=10, y=10)
text = prompt.get()

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=token) 
pipe.to(device) 

def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

trigger = ctk.CTkButton(master=app, height=40, width=120, text="CTkButton", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()
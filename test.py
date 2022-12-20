from PIL import Image
image = Image.open('./img/1.png')
new_image = image.resize((15, 15))
new_image.save('./img/New_1.png', quality=70)

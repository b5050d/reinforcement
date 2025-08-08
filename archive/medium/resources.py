import os

# Set up the Resources needed
currdir = os.path.dirname(__file__)
resource_folder = os.path.join(os.path.dirname(currdir), "resources")
sprite_path = os.path.join(resource_folder, "sprite.png")
assert os.path.exists(sprite_path)

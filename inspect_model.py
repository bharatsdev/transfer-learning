from keras.applications import VGG16

# loading the network.

print('[INFO] : Loading Network.... !')
model = VGG16(weights='imagenet', include_top=False)

print('[INFO] : Showing layers....!')


# loop over the layers in the network and display them to the console
for (idx, layer) in enumerate(model.layers):
    print('[INFO] : {} \t {}'.format(idx, layer.__class__.__name__))

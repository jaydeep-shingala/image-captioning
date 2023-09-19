# image-captioning
Image captioning using ViT for image feature extractor and then Attention mechanism for generating text description of those images
In this first, used Vision transformer to extract visual features from images. Flicker8k dataset has been used. Due to computational resources limitations 5k images are used for training purpose. First, use pretrained vision vision transformer on ImageNet and then use that pretrained model for extracting featue tensors for each image.

Then vanila Attention encoder and decoder stacks are used.
1. the image features are inputs for encoder and then more deeper self attended features are learned.
2. Next text encodings of training image captions are given as input to decoder and their attentioned features are also learned.
3. Now in decoder stack, for key, query and value, key and value comes from encoder part where image features are learned and query comes from decoder part where captions text features are larned. Now we try to decode and predict next token such that cross entropy loss is minimised.

#### The pretrained models can be found in pretrained_models.txt for both ViT and Proposed-image-captioning approach.

### For implementation and sub-model architectures, i have refered to this two novel papers and their methods.
1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: [https://arxiv.org/pdf/2010.11929.pdf]
2. Attention Is All You Need: [https://arxiv.org/pdf/1706.03762.pdf]

for more can contact at: jaydeeps@iisc.ac.in


##### *This approach to combine ViT and text attention, i could not find anywhere in the literature as of on August 31, 2023. 

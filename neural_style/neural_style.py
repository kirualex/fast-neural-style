import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformer = TransformerNet()
    if args.fine_tune:
        transformer.load_state_dict(torch.load(args.checkpoint_name + ".t7"))
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.repeat(args.batch_size, 1, 1, 1)
    style = utils.preprocess_batch(style)
    if args.cuda:
        style = style.cuda()
    style_v = Variable(style, volatile=True)
    style_v = utils.subtract_imagenet_mean_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_stable_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            xn = utils.add_noise(x)
            
            if args.cuda:
                x = x.cuda()
                xn = xn.cuda()

            y = transformer(x)
            yn = transformer(xn)
            yn = Variable(yn.data.clone(), volatile=True)
            xc = Variable(x.data.clone(), volatile=True)

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)
            
            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])
                
            tv_regularizer = utils.total_variation(y)
            stabilizing_regularizer = mse_loss(y, yn)
            total_loss = (content_loss
                          + style_loss
                          + args.lambda_tv * tv_regularizer
                          + args.lambda_sr * stabilizing_regularizer)
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            agg_stable_loss += stabilizing_regularizer.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tstability: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  agg_stable_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss + agg_stable_loss) / (batch_id + 1)
                )
                print(mesg)
            
            if (batch_id + 1) % (args.checkpoint_interval) == 0:
                transformer.eval()
                transformer.cpu()
                save_model_filename = args.checkpoint_name + ".t7"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(transformer.state_dict(), save_model_path)
                print("\nCheckpointed model saved at", save_model_path)
                transformer.train()
                if args.cuda:
                    transformer.cuda()

    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def stylize(args):
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))
    
    if args.cuda:
            style_model.cuda()
    
    if args.video:
        frames = os.listdir(args.content_path)
        
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        for i,frame in enumerate(frames):
            if not os.path.exists(args.output_path+'/'+frame):
                content_image = utils.tensor_load_rgbimage(args.content_path+'/'+frame, scale=args.content_scale)
                content_image = content_image.unsqueeze(0)

                if args.cuda:
                    content_image = content_image.cuda()
                content_image = Variable(utils.preprocess_batch(content_image), volatile=True)

                output = style_model(content_image)
                utils.tensor_save_bgrimage(output.data[0], args.output_path+'/'+frame, args.cuda)
            print ('\r|'+str(i)+'|')
        
    else:
        content_image = utils.tensor_load_rgbimage(args.content_path, scale=args.content_scale)
        content_image = content_image.unsqueeze(0)

        if args.cuda:
            content_image = content_image.cuda()
        content_image = Variable(utils.preprocess_batch(content_image), volatile=True)

        if args.cuda:
            style_model.cuda()

        output = style_model(content_image)
        utils.tensor_save_bgrimage(output.data[0], args.output_path, args.cuda)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--fine-tune", action='store_true',
                                  help="for fine-tuning a trained model")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
                                  help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--style-weight", type=float, default=5.0,
                                  help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lambda-tv", type=float, default=10e-6,
                                  help="strength of total variation regularizer, default is 10e-6")
    train_arg_parser.add_argument("--lambda-sr", type=float, default=1000.0,
                                  help="strength of stabilizing regularizer, default is 1000.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-name", type=str, default="checkpoint",
                                  help="name of checkpoint model")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="save a checkpoint every n")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--video", type=int, default=0,
                                 help="if you are stylizing video, set to 1, default is 0")
    eval_arg_parser.add_argument("--content-path", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content")
    eval_arg_parser.add_argument("--output-path", type=str, required=True,
                                 help="path for saving the output")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()

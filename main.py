import argparse

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


from data import *
from model import *

parser = argparse.ArgumentParser(description='Voxel-based Crystal NN')

parser.add_argument('-r', '--root-dir', default='voxel_data', type=str, metavar='N',
                    help='root dirctory to data set')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--train-ratio', default=0.6, type=float, metavar='N',
                    help='percentage of training data to be loaded (default 0.8)')
parser.add_argument('--val-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
parser.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')


args = parser.parse_args(sys.argv[1:])
args.cuda = not args.disable_cuda and torch.cuda.is_available()
best_mae_error = 1e10


def main():
    global args, best_mae_error
	
    model = generate_model(18)
    
    if args.cuda:
        model.cuda()
    

    dataset = Voxel_Data(args.root_dir)

    train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            return_test=True)

    sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), 0.001,
                                   weight_decay=0)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(args.epochs):
        # train for one epoch
        train_epoch(epoch, train_loader, model, criterion, optimizer, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

       # scheduler.step()

       # if mae_error < best_mae_error:
       #     best_mae_error = min(mae_error, best_mae_error)
       #     torch.save({
       #         'epoch': epoch + 1,
       #         'state_dict': model.state_dict(),
       #         'best_mae_error': best_mae_error,
       #         'optimizer': optimizer.state_dict()
       #     }, 'model_best_resnet.pth.tar')


        # test best model
      #  print('---------Evaluate Model on Test Set---------------')
      #  best_checkpoint = torch.load('model_best_resnet.pth.tar')
      #  model.load_state_dict(best_checkpoint['state_dict'])
        validate(test_loader, model, criterion, normalizer)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def train_epoch(epoch, data_loader, model, criterion, optimizer, normalizer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        inputs = inputs.transpose(1,4).float()
        targets = targets.float()
        targets_norm = normalizer.norm(targets)
        if args.cuda:
            input_var = Variable(inputs.cuda(non_blocking=True))
            targets_var = Variable(targets_norm.cuda(non_blocking=True))
        else:
            input_var = Variable(inputs)
            targets_var = Variable(targets_norm)

        
        outputs = model(input_var)
        loss = criterion(outputs, targets_var)
                
        losses.update(loss.data, inputs.size(0))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses
                      ))

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)    
    
    
def validate(data_loader, model, criterion, normalizer):
    
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        inputs = inputs.transpose(1,4).float() 
        targets = targets.float()
        target_normed = normalizer.norm(targets)
        if args.cuda:
            input_var = Variable(inputs.cuda(non_blocking=True))
            targets_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            input_var = Variable(input_var)
            targets_var = Variable(target_normed)

        outputs = model(input_var)
        loss = criterion(outputs, targets_var)
                
        losses.update(loss.data, inputs.size(0))
        
        loss.backward()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if i % 10 ==0:
            print(
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses
                      ))
    return losses.avg

if __name__ == '__main__':
    main()

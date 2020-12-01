# centerX　用中国特色社会主义的方式打开centernet

## 太长不看版
　笔者重构了一版centernet(objects as points)的代码，并加入了蒸馏，多模型蒸馏，转caffe，转onnx，转tensorRT，把后处理也做到了网络前向当中，对落地非常的友好。

　不敢兴趣的可以收藏一下笔者的表情包，如果觉得表情包好玩，跪求去github点赞。

## 前言
　centernet是我最喜欢的检测文章之一，没有anchor，没有nms，结构简单，可拓展性强，最主要的是：落地极其方便，选一个简单的backbone，可以没有bug的转成你想要的模型（caffe，onnx，tensorRT）。并且后处理也极其的方便。

　但是centernet原版的代码我初看时有点吃力，但也没有重构的想法，过了一些时日后我找到了centernet-better和centernet-better-plus，于是把他们的代码白嫖了过来然后自己完善一下，形成对我友好的代码风格。（当然剽窃最多的其实是fast reID）
  
　由于本人不喜欢写纯技术方面的博客，也不想写成一篇纯PR稿（从本科开始就深恶痛觉写实验报告），更不想让人觉得读这篇文章是在学习，所以本篇文章不太正经，也没有捧一踩一的操作，跟别人的宣传稿不太一样。毕竟代码写的不是打打杀杀，而是人情世故，真学东西还得看其他人的文章，看我的也就图一乐。

## 宣传部分
　一般来说读文章的人点进来都会带着这样一个心理，我为什么要用centerX，明明我用别的框架用的很顺利了，转过来多麻烦你知道吗，你在教我做事？

- 如果你需要用检测算法快速的落地，需要一个速度快并精度尚可的模型，而且可以无坑转caffe，onnx，tensorRT，同时基本不用写后处理，那centerX会很适合你。（原本centernet的后处理需要额外的3X3 pooling和topK的操作，被作者用一个极骚操作放到了网络里面）
- 如果你想在检测的任务上体会一下模型蒸馏的快感，在baseline上无痛涨点，或者找一些detection蒸馏的灵感，可以来centerX康康。
- 如果你同时只有两个单类标注的数据集，但是你懒得去补全这两个数据集各自缺失的类别标注，你可以尝试使用centerX训练得到一个可以同时预测两类标注的检测器。
- 如果你想基于centernet做一些学术研究，你同样也可以在centerX的projects里面重构自己的代码，和centerX里面centernet的codebase并不冲突，可以快速定位bug。
- 如果你是苦逼的学生或者悲催的工具人，你可以用centerX来向上管理你的老师或者领导，因为centerX里面的mAP点不高，稍微调一下或者加点东西就可以超越本人的baseline，到时候汇报的时候可以拍着胸脯说你跑出的东西比作者高了好几个点，然后你的KPI就可以稍微有点保障了。（文章后面会给几个方向怎么跑的比作者更高）
- centerX的底层框架白嫖自优秀检测框架detectron2，如果之前有跑过detectron2的经验，相信可以和马大师的闪电连五鞭一样，无缝衔接的使用。
- 如果没有detectron2的使用经验，那也没有关系，我专门写了懒人傻瓜式run.sh，只需要改改config和运行指令就可以愉快地跑起来了。
- 如果上述的理由都没有打动你，那么如果我用这篇文章把你逗乐了，恳求去github给个star吧。


## 代码核心思想

　受到老领导道家思维编程的启发，centerX的trick里面也贯彻了一些具有中国特色社会主义的中心主题思想。

- 代码cv大法————拿来主义
- 模型蒸馏————先富带动后富
- 多模型蒸馏，两个单类检测模型融合成为一个多类检测模型————圣人无常师
- 共产主义loss，解决模型对lr太过敏感问题————马克思主义
- 把后处理放到神经网络中————团结我们真正的朋友，以攻击我们的真正的敌人，分清敌我。《毛选》

## centerX各个模块

### 基础实现
　
　这个方面没有什么好说的，也没有做到和其他框架的差异化，只是在detectron2上对基础的centernet进行了复现而已，而且大部分代码都是白嫖自centernet-better和centernet-better-plus，就直接上在COCO上的实验结果吧。

Backbone ResNet-50

| Code             | mAP  | epoch |
| ---------------- | ---- | ----- |
| centerX          | 33.2 |  70   |
| centerX          | 34.3 |  140  |
| centernet-better | 34.9 |  140  |

Backbone ResNet-18

centerX_KD是用27.9的resnet18作为学生网络，33.2的resnet50作为老师网络蒸馏得到的结果，详细过程在在下面的章节会讲。

| Code             | mAP  | epoch |
| ---------------- | ---- | ----- |
| centerX          | 30.2 |  140  |
| centerX          | 27.9 |  70   |
| centerX_KD       | 31.0 |  70   |
| centernet-better | 29.8 |  140  |
| centernet        | 28.1 |  140  |

### 模型蒸馏

　大嘎好，我是detection。我时常羡慕的看着隔壁村的classification，embedding等玩伴，他们在蒸馏上面都混得风生水起，什么logits蒸馏，什么KL散度，什么Overhaul of Feature Distillation。每天都有不同的家庭教师来指导他们，凭什么我detection的教育资源就很少，我detection什么时候才能站起来!

　造成上述的原因主要是因为detection的范式比较复杂，并不像隔壁村的classification embedding等任务，整张图输入后输出一个vector：

- 1.two stage的网络本身由于rpn输出的不确定性，导致teacher和student的proposal对齐是个大问题。
- 2.笔者尝试过在中间层feature上进行蒸馏，这样就可以偷懒不用写最后的logits蒸馏部分的代码了，结果基本没有啥作用，还是得在logits上蒸馏比较稳。
- 3.我编不下去了

　我们再来回头看看centernet的范式，哦，我的上帝，多么简单明了的范式：

- 1.网络输出三个头，一个预测中心点，一个预测宽高，一个预测中心点的偏移量
- 2.没有复杂的正负样本采样，只有物体的中心点是正样本，其他都是负样本

　这让笔者看到了在detection上安排家庭教师的希望，于是我们仿照了centernet本来的loss的写法，仿照了一个蒸馏的loss。具体的实现可以去code里面看，这里就说一下简单的思想。

- 1.对于输出中心点的head，把teacher和student输出的head feature map过一个relu层，把负数去掉，然后做一个mse的loss，就OK了。
- 2.对于输出宽高和中心点的head，按照原centernet的实现是只学习正样本，在这里笔者拍脑袋想了一个实现方式：我们用teacher输出中心点的head过了relu之后的feature作为系数，在宽高和中心点的head上所有像素点都做L1 loss后和前面的系数相乘。
- 3.在蒸馏时，三个head的蒸馏loss差异很大，需要手动调一下各自的loss weight，一般在300次迭代后各个蒸馏loss在0~3之间会比较好。
- 4.所以在之前我都是300次epoch之后直接停掉，然后根据当前loss 预估一个loss weight重新开始训练。这个愚蠢的操作在我拍了另外一次脑袋想出共产主义loss之后得以丢弃。
- 5.在模型蒸馏时我们既可以在有标签的数据上联合label的loss进行训练，也可以直接用老师网络的输出在无标签的数据集上蒸馏训练。基于这个特性我们有很多妙用
- 6.当在有标签的数据上联合label的loss进行训练时，老师训N个epoch，学生训N个epoch，然后老师教学生，并保留原本的label loss再训练N个epoch，这样学生的mAP是训出来最高的。
- 7.当在无标签的数据集上蒸馏训练时，我们就跳出了数据集的限制，先在有标签的数据集上老师训N个epoch，然后老师在无标签的数据集上蒸馏学生模型训练N个epoch，可以使得学生模型的精度比baseline要高，并且泛化性能更好。
- 8.之前在centernet本来的code上还跑过一个实验，相同的网络，自己蒸馏自己也是可以涨点的。在centerX上我忘记加进去了。

　我们拉到实验的部分。

| Backbone                 | crowd mAP  | coco_person mAP  | epoch | teacher | student_pretrain | train_set |
| ----------------         | ----       | --------------   | ----- |  -----  | -------          | -----     |
| resdcn50                 | **35.1**   |    35.7          |  80   |   -     |    -             | crowd     |
| resdcn18(baseline)       | 31.2       |    31.2          |  80   |   -     |    -             | crowd     |
| resdcn18_KD              | 34.5       |    34.9          |  80   | resdcn50| resdcn18         | crowd     |
| resdcn18_KD_woGT_scratch | 32.8       |    34.2          |  140  | resdcn50| imagenet         | crowd     |
| resdcn18_KD_woGT_scratch | 34.1       |  **36.3**        |  140  | resdcn50| imagenet         | crowd+coco|

### 多模型蒸馏
　
　看到蒸馏效果还可以，可以在不增加计算量的情况下无痛涨点，笔者高兴了好一阵子，直到笔者在实际项目场景上遇到了一个尴尬地问题：
- 我有一个数据集A，里面有物体A的标注
- 我有一个数据集B，里面有物体B的标注
- 现在由于资源有限，只能跑一个检测网络，我怎么得到可以同时预测物体A和物体B的检测器？

　因为数据集A里面可能会有大量的未标注的B，B里面也会有大量的未标注的A，直接放到一起训练肯定不行，网络会学傻。
- 常规的操作是去数据集A里面标B，然后去数据集B里面标A，这样在加起来的数据集上就可以训练了。但是标注成本又很贵，这让洒家如何是好？
- 稍微骚一点的操作是在A和B上训练两个网络，然后在缺失的标注数据集上预测伪标签，然后在补全的数据集上训练
- novelty更高的操作是在没有标注的数据集上屏蔽网络对应的输出，（该操作仅在C个二分类输出的检测器下可用）
- 有没有一种方法，也不用标数据，也不用像伪标签那么粗糙，直接躺平，同时novelty也比较高，比较好跟领导说KPI的一个方法？

　在笔者再次拍了拍脑袋后，发挥了我最擅长的技能：白嫖。想到了这样一个方案：

- 我先在数据A上训练个老师模型A，然后在数据B上训练老师模型B，然后我把老师模型A和B的功力全部传给学生模型C，岂不美哉？
- 我们再来看看centernet的范式，我再次吹爆这个作者的工作，不仅简单易懂的支持了centerPose，centertrack，center3Ddetection，还可以输出可旋转的物体检测。
- 无独有偶，可能是为了方便复用focal loss，作者在分类时使用了C个二分类的分类器，而不是softmax分类，这给了笔者白嫖的灵感：既然是C个二分类的分类器，那么对于每一个类别，那么我们可以给学生网络分别找一个家庭教师，这样就可以拥有双倍的快乐。

  那么我们的多模型蒸馏就可以用现有的方案拼凑起来了。这相当于我同时白嫖了自己的代码，以及不完整标注的数据集，白嫖是真的让人快乐啊。和上述提到的操作进行一番比♂较，果然用了圣人无常师的多模型蒸馏的效果要好一些。笔者分别在人体和车，以及人体和人脸上做了实验。 数据集为coco_car,crowd_human,widerface.

| Backbone                 |  mAP crowd     |  mAP coco_car  | epoch | teacher | student_pretrain | train_set |
| ----------------         | ----           | ----           | ----- |  -----  | -------          | -----     |
| 1.resdcn50               | 35.1           | -              |  80   |   -     |    -             | crowd     |
| 2.resdcn18               | 31.7           | -              |  70   |   -     |    -             | crowd     |
| 3.resdcn50               | -              | 31.6           |  70   |   -     |    -             | coco_car  |
| 4.resdcn18               | -              | 27.8           |  70   |   -     |    -             | coco_car  |
| resdcn18_KD_woGT_scratch | 31.6           | 29.4           |  140  |  1,3    | imagenet         | crowd+coco_car|

| Backbone                 |  mAP crowd_human |  mAP widerface | epoch | teacher | student_pretrain | train_set |
| ----------------         | ----             | -------------- | ----- |  -----  | -------          | -----     |
| 1.resdcn50               | 35.1             | -              |  80   |   -     |    -             | crowd     |
| 2.resdcn18               | 31.7             | -              |  70   |   -     |    -             | crowd     |
| 3.resdcn50               | -                | 32.9           |  70   |   -     |    -             | widerface |
| 4.resdcn18               | -                | 29.6           |  70   |   -     |    -             | widerface |
| 5.resdcn18_ignore_nolabel| 29.1             | 24.2           |  140  |   -     |    -             |crowd+wider|
| 6.resdcn18_pseudo_label  | 28.9             | 27.7           |  140  |   -     |    -             |crowd+wider|
| 7.resdcn18_KD_woGT_scratch| 31.3            | 32.1           |  140  |  1,3    | imagenet         |crowd+wider|


### 共产主义loss

　笔者在训练centerX时，出现过这样一个问题，设置合适的lr时，训练的一切都那么自然又和谐，而当我lr设置大了以后，有时候会训到一半，网络直接loss飞涨然后mAP归零又重新开始往上爬，导致最后模型的mAP很拉胯。对于这种情况脾气暴躁的我直接爆了句粗口：给老子爬。 骂完了爽归爽，问题还是要解决的，为了解决这个问题，笔者首先想到笔者的代码是不是哪里有bug，但是找了半天都没找到，笔者还尝试了如下的方式：

  - 加入clip gradients，不work
  - 自己加了个skip loss，当本次iter的loss是上次loss的k=1.1倍以上时，这次loss全部置0，不更新网络，不work
  - 换lr_scheduler, 换optimalizer， 不work

　看来这个bug油盐不进，软硬不吃。总会出现某个时间段loss突然增大，然后网络全部从头开始训练的情况。这让我想到了内卷加速，资本主义泡沫破裂，经济大危机后一切推倒重来。这个时候才想起共产主义的好，毛主席真是永远滴神。既然如此，咱们一不做二不休，直接把蛋糕给loss们分好，让共产主义无产阶级的光照耀到它们身上，笔者一气之下把loss大小给各个兔崽子head们给规定死，具体操作如下：

  - 给每个loss设置一个可变化的loss weight，让loss一直保持在一个固定的值。
  - 考虑到固定的loss值比较硬核，笔者把lr设置为cosine的lr，让lr比较平滑的下降，来模拟正常情况下网络学习到的梯度分布。
  - 其实本loss可以改名叫adaptive loss，但是为了纪念这次的气急败坏和思维升华，笔者依然任性的把它称之为共产主义loss。

　接下来就是实验部分看看管不管用了，于是笔者尝试了一下之前崩溃的lr，得益于共产主义的好处，换了几个数据集跑实验都没有出现mAP拉胯的情况了，期间有几次出现了loss飞涨的情况，但是在共产主义loss强大的调控能力之下迅速恢复到正常状态，看来社会主义确实优越。同时笔者也尝试了用合适的lr，跑baseline和共产主义loss的实验，发现两者在±0.3的mAP左右，影响不大。

　笔者又为此高兴了好一段时间，并且发现了共产主义loss可以用在蒸馏当中，并且表现也比较稳定，在±0.2个mAP左右。这下蒸馏可以end2end训练了，再也不用人眼去看loss、算loss weight、停掉从头训了。

### 模型加速

　这个部分的代码都在代码的projects/speedup中，注意网络中不能包含DCN。

　centerX中提供了转caffe，转onnx的代码，onnx转tensorRT只要装好环境后一行指令就可以转换了，还提供了转换后不同框架的前向代码。
　其中笔者还找到了centernet的tensorRT前向版本（后续笔者把它称为centerRT），在里面用cuda写了centernet的后处理（包括3X3 max pool和topK后处理）。笔者转完了tensorRT之后想直接把centerRT白嫖过来，结果发现还是有些麻烦，centerRT有点像是为了centernet原始实现定制化去写的。这就有了以下的问题

  - 不仅是tensorRT版本，所有的框架上我都不想写麻烦的后处理，我想把麻烦的操作都写到网络里面去
  - 在网络中心点head的输出再加一层3X3的max pooling，可以减少一部分后处理的代码
  - 有没有办法使得最后中心点head的输出满足以下条件：1.除了中心点之外，其他的像素值全是0，（相当于已经做过了pseudo nms）；2.后处理只需要在这个feature上遍历>thresh的像素点位置就可以了。
  - 如果x1表示centernet的中心点输出，x2表示经过了3X3 maxpool之后的输出，那么在python里面其实只需要写上一行代码就得到上述的条件：y = x1[x1==x2]。但是笔者在使用转换时，onnx不支持==的操作。得另谋他路。
  
　这次笔者拍碎了脑袋都没想到怎么白嫖，于是在献祭了几根珍贵的头发之后，强行发动了甩锅技能，把后处理操作扔给神经网络，具体操作如下：
  
  - x2是x1的max pool，我们需要的是x1[x1==x2]的feature map
  - 那么我们只需要得到x1==x2,也就是一张二值化的mask就可以了。
  - 由于x2是x1的max pool，所以x1-x2 <= 0, 我们在x1-x2上加一个很小的数，使得等于0的的像素点变成正数，小于0的像素点仍然为负数。然后在加个relu，乘以一个系数使得正数缩放到1，0依然为0，就可以得到我们想要的东西了。
  
　代码如下：
```
def centerX_forward(self, x):
    x = self.normalizer(x / 255.)
    y = self._forward(x)
    fmap_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(y['cls'])
    keep = (y['cls'] - fmap_max).float() + 1e-9
    keep = nn.ReLU()(keep)
    keep = keep * 1e9
    result = y['cls'] * keep
    ret = [result,y['reg'],y['wh']]  ## change dict to list
    return ret
```

　onnx中可视化如下：
  
　值得注意的是上述骚操作在转caffe的时候会报错，所以不能加。如果非要添加上去，得在caffe的prototxt中自行添加scale层，elementwise层，relu层，这个笔者没有实现，大家感兴趣可以自行添加。

  
### 优化方向

　考虑到大家需要向上管理，笔者写几个可以涨点的东西

- 在centernet作者本来的issue里面提到，centernet很依赖于网络最后一层的特征，所以加上dlaup会涨点特别明显，但是由于feature的channel太多，会有一些时间损耗。笔者实测在某个backbone+deconv上加上dlaup之后，batchsize=8时间由32ms->44ms左右，有一些代价，所以笔者没有加。后续应该可以把dlaup里面的卷积全部改为depthwise的，找到一个速度和精度的平衡
- 想想办法看看能不能把Generalized Focal Loss，Giou loss等等剽窃过来，稍微改一下加到centernet里面
- 调参，lr，lossweight，或者共产主义loss里面各个固定loss值，每个数据集上每个backbone的参数都可以优化
- 用一个牛逼的pretrain model
- 把隔壁fast reid的自动超参搜索白嫖过来

  除了以上的在精度方面的优化之外，其实笔者还想到很多可以做的东西，咱们不在精度这个地方跟别人卷，因为卷不过别人，检测这个领域真是神仙打架，打不过打不过。我们想着把蛋糕做大，大家一起有肉吃
  
- centerPose，其实本来作者的centerpose就已经做到一个网络里面去了，但是笔者觉得可以把白嫖发挥到极致，把只在pose数据集上训过的simplebaseline蒸馏到centernet里面去，这样的好处是检测的标注和pose的标注可以分开，作为两个单独的数据集去标注，这样的话可以白嫖的数据集就更多了，并且做到一个网络里面速度会更快。
- centerPoint，直接输出矩形框四个角点相对于中心点的偏移量，而不是矩形框的宽高，这样的话相当于检测的输出是个任意四边形，这样的话我们在训练的时候可以加入任何旋转的数据增强而不用担心标注框变大的问题，同时说不定我们用已有的检测数据集+旋转数据增强训练出来的网络就具备了预测旋转物体的能力。这个网络在检测车牌，或者身份证以及发票等具有天然的优势，直接预测四个角点，不用做任何的仿射变换，也不用换成笨重的分割网络了。

## 结语

　其实有太多的东西想加到centerX里面去了，里面有很多很好玩的以及非常具有实用价值的东西都可以去做，但是个人精力有限，而且刚开始做centerX完全是基于兴趣爱好去做的，本人也只是渣硕，无法full time扑到这个东西上面去，所以上述的优化方向看看在我有生之年能不能做出来，做不出来给大家提供一个可行性思路也是极好的。

　非常感谢廖星宇，何凌霄对centerX的代码，以及发展方向上的贡献,郭聪，于万金，蒋煜襄，张建浩等同学对centerX加速模块的采坑指导。
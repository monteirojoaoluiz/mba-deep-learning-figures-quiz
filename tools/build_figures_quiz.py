from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "figures_quiz"
ASSET_DIR = OUT_DIR / "assets"
HTML_PATH = OUT_DIR / "deep_learning_figures_quiz.html"
STRONG_ANSWERS_PATH = OUT_DIR / "strong_answers.json"
DOCS_DIR = ROOT / "docs"
DOCS_ASSET_DIR = DOCS_DIR / "assets"
DOCS_HTML_PATH = DOCS_DIR / "index.html"


@dataclass(frozen=True)
class FigureSpec:
    source: str
    pdf: str
    page: int
    figure_id: str
    caption: str
    question: str
    answer: str
    image_index: int | None = None
    crop: tuple[float, float, float, float] | None = None


def topic_answer(caption: str) -> str:
    def answer(*parts: str) -> str:
        return "\n".join(f"- {part}" for part in parts)

    text = caption.lower()
    if "biological neuron" in text:
        return answer(
            "The figure shows a biological neuron: dendrites receive signals, the cell body integrates them, and the axon transmits an output signal to other neurons.",
            "Artificial neurons borrow the high-level idea of combining inputs and producing an output, but they simplify it into weighted sums, biases, and activation functions.",
            "The analogy is useful historically and conceptually, but modern neural networks are engineering models trained by optimization, not realistic simulations of brains.",
        )
    if "threshold logic" in text:
        return answer(
            "A threshold logic unit computes a weighted sum of the inputs, usually plus a bias term, then applies a step function to decide whether the unit activates.",
            "The weights determine how strongly each input contributes, while the bias shifts the decision boundary.",
            "Because the decision is based on a linear weighted sum, a single TLU can only separate linearly separable classes; nonlinear problems require feature transformations or hidden layers.",
        )
    if "perceptron" in text:
        return answer(
            "The Perceptron is a single-layer neural classifier: input features flow directly to output units through learned weights.",
            "The bias neuron lets the model shift its separating hyperplanes instead of forcing them through the origin, and multiple output neurons can represent multiple classes.",
            "Its main limitation is linearity. It can solve linearly separable classification problems, but it cannot solve XOR-like tasks without hidden layers or nonlinear feature engineering.",
        )
    if "xor" in text:
        return answer(
            "XOR places same-class points on opposite corners, so no single straight line can separate the classes.",
            "The hidden layer in the MLP creates intermediate features that transform the input into a representation where the final output neuron can separate the classes.",
            "This is the key reason depth and nonlinear activation functions matter: stacked nonlinear layers can model decision boundaries that a single linear classifier cannot.",
        )
    if "activation" in text:
        return answer(
            "Activation functions introduce nonlinearity after a layer's weighted sum. Without them, any stack of dense layers collapses mathematically into one linear transformation.",
            "Sigmoid and tanh squash values into bounded ranges but can saturate, creating very small gradients. ReLU is simple and efficient, but can output zero gradients for inactive units.",
            "Choosing activations affects optimization, gradient flow, and representational power, so it is not just a cosmetic architectural detail.",
        )
    if "softmax" in text:
        return answer(
            "The network first transforms input features through hidden layers, commonly with ReLU activations, producing increasingly abstract representations.",
            "The output layer produces logits, and softmax converts those logits into normalized probabilities that sum to one.",
            "Softmax is appropriate for single-label multiclass classification. For independent multilabel outputs, sigmoid activations per class are usually more appropriate.",
        )
    if "receptive" in text or "padding" in text or "stride" in text:
        return answer(
            "The figure concerns how CNN layers connect to local spatial regions instead of all input pixels. Each neuron sees a receptive field, not the whole image at once.",
            "Padding controls whether borders are preserved and how output size changes; stride controls how far the filter moves and therefore how much the layer downsamples.",
            "These choices trade off spatial detail, computation, and invariance. Larger strides reduce cost but can discard fine detail; padding preserves feature-map size but adds artificial border values.",
        )
    if "filters" in text or "feature map" in text or "convolutional" in text:
        return answer(
            "A convolutional layer applies learned filters across the input. Each filter produces a feature map showing where a particular pattern is detected.",
            "The same filter weights are reused at many spatial positions, which greatly reduces parameters and gives CNNs translation equivariance.",
            "Early filters often detect edges, textures, or color contrasts. Deeper layers combine these activations into object parts and task-specific semantic features.",
        )
    if "pooling" in text:
        return answer(
            "Pooling aggregates values in a local window, for example by taking the maximum or average activation.",
            "It reduces spatial resolution and computation while keeping strong evidence that a feature appeared somewhere in the local region.",
            "This adds tolerance to small translations, but excessive pooling can remove location information needed for tasks such as detection and segmentation.",
        )
    if "invariance" in text:
        return answer(
            "The figure illustrates invariance: the desired output should remain stable even if the input is shifted or slightly transformed.",
            "CNNs encourage this through local filters, weight sharing, pooling, data augmentation, and hierarchical feature learning.",
            "Some invariance is useful for classification, but dense tasks may need precise location information, so architecture design must balance robustness with spatial accuracy.",
        )
    if "typical cnn" in text or "cnn architecture" in text:
        return answer(
            "A typical CNN begins with convolutional layers that learn local visual features, often followed by nonlinear activations and pooling or strided layers that reduce resolution.",
            "As depth increases, feature maps usually become spatially smaller but semantically richer: early layers capture edges/textures, while later layers capture parts and objects.",
            "The final classifier or prediction head maps learned representations to task outputs. This is end-to-end learning because the feature extractor and predictor are optimized jointly from the training objective.",
        )
    if "inception" in text:
        return answer(
            "An Inception module sends the same input through several parallel branches, often using 1 x 1, 3 x 3, and 5 x 5 convolutions plus pooling.",
            "The branches let the network learn features at multiple spatial scales instead of committing to one kernel size everywhere.",
            "1 x 1 convolutions reduce or mix channels, which controls parameter count and computation before expensive larger convolutions.",
        )
    if "googlenet" in text:
        return answer(
            "GoogLeNet is a deep CNN built from stacked Inception modules. Its core design goal is to increase depth and representational richness without an impractical number of parameters.",
            "It uses parallel convolution paths and dimensionality reduction, especially through 1 x 1 convolutions, to combine multiscale features efficiently.",
            "Compared with a plain stack of large convolutions, it is more parameter-efficient and better able to capture patterns at different visual scales.",
        )
    if "residual" in text or "resnet" in text or "skip connection" in text:
        return answer(
            "Residual learning lets a block learn a correction F(x) that is added back to the original input x, instead of forcing the block to learn the whole transformation from scratch.",
            "Skip connections improve gradient flow during backpropagation, which helps very deep networks train without degradation.",
            "When feature-map size or channel depth changes, the shortcut path must be adapted, commonly with projection or strided convolution, so the tensors can be added.",
        )
    if "depthwise" in text or "separable" in text:
        return answer(
            "Depthwise separable convolution splits a standard convolution into two operations: depthwise spatial filtering per input channel, followed by pointwise 1 x 1 convolution to mix channels.",
            "This reduces parameters and multiply-add operations because spatial filtering and channel mixing are no longer done simultaneously for every output channel.",
            "It is especially useful in efficient architectures for mobile or low-latency vision systems, though it may reduce capacity if used too aggressively.",
        )
    if "se-" in text or "squeeze" in text or "recalibration" in text:
        return answer(
            "A Squeeze-and-Excitation block learns channel-wise attention. It first summarizes each feature map, usually with global average pooling, then predicts weights for the channels.",
            "Those weights recalibrate the feature maps, amplifying informative channels and suppressing less useful ones.",
            "This improves representational selectivity with modest extra computation, because it changes channel importance without changing the spatial layout.",
        )
    if "iou" in text or "bounding boxes" in text:
        return answer(
            "Intersection over Union is the area of overlap between predicted and ground-truth boxes divided by the area covered by their union.",
            "It measures localization quality: a high IoU means the predicted box closely matches the target object, while a low IoU means poor overlap.",
            "Object detection systems use IoU for evaluation, assigning positives/negatives, and non-maximum suppression. Raising the IoU threshold makes evaluation stricter.",
        )
    if "sliding" in text or "multiple objects" in text:
        return answer(
            "The figure shows object detection over many spatial locations rather than a single image-level classification decision.",
            "Each grid cell or sliding location can predict objectness, class scores, and bounding-box coordinates, allowing multiple objects to be detected in one image.",
            "The task is harder than classification because the model must solve both recognition and localization, handle multiple objects, and suppress duplicate detections.",
        )
    if "fully convolutional" in text:
        return answer(
            "A fully convolutional network replaces dense layers with convolutional layers, so the model preserves a spatial grid of predictions.",
            "Because convolution does not require a fixed input size in the same way dense layers do, the same model can process larger images and produce correspondingly larger feature maps.",
            "This is important for dense prediction tasks such as segmentation and detection, where the output must retain spatial correspondence with the input.",
        )
    if "semantic segmentation" in text:
        return answer(
            "Semantic segmentation assigns a class label to every pixel, such as road, person, car, or building.",
            "It differs from classification because the output is not one label for the whole image, and it differs from object detection because it predicts dense pixel regions rather than bounding boxes.",
            "Good segmentation requires both semantic understanding and spatial precision, which is why architectures often combine deep features with upsampling and skip connections.",
        )
    if "transposed" in text or "upsampling" in text or "skip layers" in text:
        return answer(
            "Upsampling increases the spatial resolution of feature maps so a network can produce dense outputs aligned with the input image.",
            "Transposed convolutions can learn how to upsample, while simpler interpolation-based methods can be fixed and followed by convolution.",
            "Skip layers combine high-level semantic features with lower-level spatial details, improving boundaries and fine structures in segmentation outputs.",
        )
    if "recurrent neuron" in text or "unrolled" in text:
        return answer(
            "A recurrent neuron or layer processes a sequence by reusing the same parameters at each time step and carrying hidden state forward.",
            "Unrolling through time shows the repeated computation as if it were a deep network stretched across sequence positions.",
            "This makes RNNs suitable for ordered data such as text, audio, and time series, but long sequences can cause vanishing or exploding gradients.",
        )
    if "hidden state" in text:
        return answer(
            "The hidden state is the recurrent memory passed from one time step to the next. It lets the model condition current predictions on previous inputs.",
            "The output is what the model exposes at a time step, while the hidden state can contain internal information that is not directly emitted.",
            "Separating state from output is useful when the model needs to remember context but only produce selected predictions, such as sequence classification or encoder-decoder models.",
        )
    if "seq-to-seq" in text or "vector-to-seq" in text or "encoder" in text:
        return answer(
            "Sequence-to-sequence models produce an output at each input step, useful for tasks like time-series prediction or part-of-speech tagging.",
            "Sequence-to-vector models summarize a whole sequence into one output, such as sentiment classification. Vector-to-sequence models generate sequences from a fixed input, such as image captioning.",
            "Encoder-decoder models first encode an input sequence into a representation, then decode an output sequence, which is useful for translation and other transduction tasks.",
        )
    if "backpropagation through time" in text:
        return answer(
            "Backpropagation through time trains an RNN by unrolling it across time steps, computing losses, and propagating gradients backward through the unrolled computation graph.",
            "The same recurrent weights are reused at each step, so their gradients accumulate contributions from many positions in the sequence.",
            "Long unrolls can create vanishing/exploding gradients; common mitigations include gated cells, gradient clipping, shorter truncated BPTT windows, and normalization or careful initialization.",
        )
    if "time series" in text or "forecast" in text:
        return answer(
            "Time-series forecasting uses previous observations to predict future values while respecting temporal order.",
            "Inputs are usually windows of past time steps, and targets may be the next value, several future values, or a shifted sequence.",
            "For multi-step forecasting, direct prediction avoids recursive error accumulation, while iterative one-step forecasting is simple but can compound mistakes over the horizon.",
        )
    if "deep rnn" in text:
        return answer(
            "A deep RNN stacks recurrent layers vertically. Each time step passes information upward through layers and forward through time.",
            "Lower layers can model local temporal patterns, while higher layers can combine them into more abstract sequence representations.",
            "Depth increases capacity but also makes optimization harder, so gated cells, dropout between layers, clipping, and careful validation become more important.",
        )
    if "lstm" in text:
        return answer(
            "An LSTM cell maintains a cell state that acts as a long-term memory path, controlled by gates.",
            "The forget gate decides what old information to keep, the input gate controls what new information to write, and the output gate controls what part of the state is exposed.",
            "This gated additive memory path helps gradients travel over longer time spans, making LSTMs better than simple RNNs for long-range dependencies.",
        )
    if "gru" in text:
        return answer(
            "A GRU is a gated recurrent unit that simplifies the LSTM idea by using fewer gates and no separate cell state.",
            "The update gate controls how much previous state is retained, while the reset gate controls how much past information is used when computing the candidate state.",
            "GRUs often train faster and use fewer parameters than LSTMs, while still handling longer dependencies better than a plain RNN.",
        )
    if "wavenet" in text:
        return answer(
            "WaveNet models sequences with causal convolutions, meaning predictions at a time step only depend on current and previous inputs, not future values.",
            "Dilated convolutions skip over increasing gaps, giving the network a large temporal receptive field without requiring recurrent loops.",
            "Compared with RNNs, this can be more parallelizable during training and can capture long-range patterns efficiently, but architecture and receptive-field design become critical.",
        )
    if "clip" in text or "vision language" in text:
        return answer(
            "A vision-language model connects visual inputs and language by mapping images and text into compatible representations.",
            "In CLIP-style training, an image encoder and text encoder are trained so matching image-text pairs are close in embedding space and mismatched pairs are far apart.",
            "This enables zero-shot classification, image search, caption-based retrieval, visual question answering, and multimodal assistants, but performance depends heavily on training data coverage and prompt design.",
        )
    if "graph" in text or "node" in text:
        return answer(
            "Graph learning uses nodes, edges, attributes, and neighborhood structure rather than fixed grids or sequences.",
            "Node embeddings map graph entities into vectors that preserve useful relational information, so downstream models can perform node classification, link prediction, clustering, or graph classification.",
            "Graphs are challenging because they have irregular topology, variable node degrees, no natural ordering, and dependencies that can extend over many hops.",
        )
    if "traditional" in text or "end-to-end" in text:
        return answer(
            "Traditional ML pipelines often rely on manually designed features followed by a separate classifier or regressor, while deep learning learns feature representations and predictors jointly.",
            "End-to-end learning means optimizing the whole pipeline from raw or minimally processed input to final output using a single training objective.",
            "Its advantages are representation learning and reduced manual feature engineering, especially for images, audio, text, and other high-dimensional data. Its disadvantages include higher data needs, compute cost, lower interpretability, and risk of overfitting or shortcut learning.",
        )
    return answer(
        "The figure should be identified by naming the model component, task, or training mechanism it represents.",
        "A strong answer should explain the information flow: what enters the component, what transformation occurs, and what output is produced.",
        "The concept matters if it improves representation learning, optimization, generalization, computational efficiency, or suitability for a specific data type such as images, sequences, graphs, or multimodal inputs.",
    )


def make_question(caption: str) -> str:
    text = caption.lower()
    if "biological neuron" in text or "biological neural network" in text:
        return "- Use the figure to explain which biological ideas inspired artificial neural networks.\n- Which parts of the biological system have loose analogues in artificial neurons or layers?\n- Why should this analogy not be taken too literally?"
    if "logical computations" in text:
        return "- Name the logical operations represented in the figure.\n- Explain how artificial neurons can implement simple logical rules.\n- What limitation appears when a problem is not linearly separable?"
    if "threshold logic" in text:
        return "- Name and briefly describe the artificial neuron shown in the figure.\n- Explain the role of weights, inputs, bias, and threshold/step activation.\n- Why is this unit only able to learn linear decision boundaries?"
    if "perceptron" in text:
        return "- Describe the Perceptron architecture shown in the figure.\n- What is the role of the bias neuron and the output neurons?\n- For which type of classification problem is this architecture suitable, and where does it fail?"
    if "xor" in text:
        return "- Explain why the XOR problem cannot be solved by a single linear classifier.\n- Use the MLP in the figure to explain how a hidden layer changes the representation.\n- What does this example show about the value of depth and nonlinear activations?"
    if "multilayer perceptron" in text:
        return "- Identify the input, hidden, and output layers in the figure.\n- Explain how information flows through a multilayer perceptron.\n- Why do hidden layers and nonlinear activations make the model more expressive than a Perceptron?"
    if "activation functions" in text:
        return "- Name the activation functions shown in the figure and compare their shapes.\n- Why do neural networks need nonlinear activation functions?\n- What practical issues can arise from saturating or zero-gradient regions?"
    if "softmax" in text or "modern mlp" in text:
        return "- Describe the role of each layer in the classification network shown in the figure.\n- Why is ReLU commonly used in hidden layers?\n- What does the softmax layer output, and when is it appropriate?"
    if "receptive fields" in text:
        return "- Name the CNN idea illustrated by the local receptive fields.\n- Why do CNNs connect neurons to local image regions instead of every input pixel?\n- How does this design help with images compared with a fully connected network?"
    if "zero padding" in text or "padding" in text:
        return "- Explain the convolution operation illustrated in the figure.\n- What is zero padding, and how does it affect the output feature-map size?\n- Why might a CNN designer choose padding instead of shrinking the feature maps at every layer?"
    if "stride" in text:
        return "- Explain how stride changes the convolution or pooling operation shown in the figure.\n- What happens to spatial resolution when stride increases?\n- What tradeoff does stride create between computation, detail, and invariance?"
    if "filters" in text or "feature maps" in text:
        return "- Which CNN layer type is illustrated by the filters and feature maps?\n- What does a learned filter detect, and why are learned filters preferable to fixed hand-designed filters?\n- How do feature maps contribute to deeper CNN representations?"
    if "multiple feature maps" in text or "three color channels" in text:
        return "- Explain how convolution works when the input has multiple channels.\n- What is the relationship between filters, channels, and output feature maps?\n- Why do deeper CNN layers usually contain many feature maps?"
    if "pooling" in text:
        return "- Name the CNN layer shown in the figure and describe how it operates.\n- What is the purpose of pooling layers in a CNN architecture?\n- What information can be lost when pooling is used too aggressively?"
    if "invariance" in text:
        return "- Use the figure to explain the concept of invariance in computer vision.\n- Why is some invariance useful for image classification?\n- When could too much invariance be harmful for a vision task?"
    if "typical cnn architecture" in text:
        return "- Briefly explain the structure and role of the CNN layers depicted in the figure.\n- How do convolution and pooling layers transform raw pixels into useful features?\n- Use the figure to explain the concept of end-to-end learning."
    if "generating new training" in text:
        return "- What data-processing technique is shown in the figure?\n- Why does data augmentation help deep learning models generalize?\n- Give examples of image transformations that are useful, and one transformation that could change the label."
    if "inception" in text:
        return "- Describe the structure of the Inception module shown in the figure.\n- Why does it use parallel branches with different kernel sizes?\n- How do 1 x 1 convolutions help control computation and parameters?"
    if "googlenet" in text:
        return "- Explain the high-level architecture shown in the figure.\n- How do Inception modules make GoogLeNet deeper while remaining parameter-efficient?\n- What problem was this architecture trying to solve compared with simpler CNN stacks?"
    if "residual learning" in text or "deep residual" in text or "resnet" in text or "skip connection" in text:
        return "- Use the figure to explain residual learning or skip connections.\n- Why do skip connections help train very deep neural networks?\n- What must be handled when the input and output dimensions of a residual block differ?"
    if "depthwise separable" in text:
        return "- Describe the two stages of the depthwise separable convolution shown in the figure.\n- How does it differ from a standard convolution?\n- Why is this operation useful for efficient CNN architectures?"
    if "se-" in text or "squeeze" in text or "recalibration" in text:
        return "- Explain what the SE block/module is doing in the figure.\n- How does feature-map recalibration change the information passed forward?\n- Why can channel attention improve a CNN without changing the spatial resolution?"
    if "iou" in text:
        return "- Define Intersection over Union using the boxes shown in the figure.\n- Why is IoU useful for object detection?\n- How would changing the IoU threshold affect precision and recall?"
    if "sliding a cnn" in text or "multiple objects" in text:
        return "- Explain how the figure detects multiple objects in one image.\n- What does each spatial grid location or sliding-window prediction need to output?\n- Why is this harder than image classification?"
    if "fully convolutional" in text:
        return "- What makes the network in the figure fully convolutional?\n- Why can it process different image sizes?\n- How does this idea support dense prediction tasks such as detection or segmentation?"
    if "semantic segmentation" in text:
        return "- Name the computer-vision task shown in the figure.\n- How is semantic segmentation different from classification and object detection?\n- Why is preserving spatial detail important for this task?"
    if "transposed" in text or "upsampling" in text or "skip layers" in text:
        return "- Explain the upsampling or skip-layer operation shown in the figure.\n- Why do segmentation networks need to recover spatial resolution?\n- How do skip connections from lower layers improve dense predictions?"
    if "recurrent neuron" in text or "unrolled" in text:
        return "- Explain what makes the neuron or layer in the figure recurrent.\n- What does it mean to unroll an RNN through time?\n- Why are RNNs suitable for sequence data, and what limitation do they have?"
    if "hidden state" in text:
        return "- Use the figure to distinguish hidden state from output.\n- Why is hidden state useful when processing sequences?\n- Give an example task where the model should keep internal information that is not directly emitted at every time step."
    if "seq-to-seq" in text or "seq-to-vector" in text or "vector-to-seq" in text or "encoder" in text:
        return "- Name the four sequence architectures shown in the figure.\n- Give one application for each architecture.\n- Why is an encoder-decoder design useful for tasks such as translation?"
    if "backpropagation through time" in text:
        return "- Explain the training procedure illustrated in the figure.\n- Why can an unrolled RNN be trained with backpropagation?\n- What problem can occur when gradients must pass through many time steps, and how can it be mitigated?"
    if "time series" in text:
        return "- Describe the forecasting setup shown in the figure.\n- What are the inputs and targets in a time-series forecasting problem?\n- How is sequence forecasting different from ordinary tabular prediction?"
    if "forecasting 10 steps" in text:
        return "- Explain the forecasting strategy shown in the figure.\n- What is the difference between predicting many future steps directly and predicting one step at a time repeatedly?\n- What error-propagation risk appears in iterative forecasting?"
    if "deep rnn" in text:
        return "- Describe the deep RNN architecture shown in the figure.\n- Why might stacking recurrent layers improve sequence modeling?\n- What training or optimization issues become more important as the RNN gets deeper?"
    if "lstm" in text:
        return "- Name the recurrent cell shown in the figure and identify its main gates/state paths.\n- How does this cell address the vanishing-gradient problem?\n- In what kinds of sequence tasks would you prefer this cell over a simple RNN?"
    if "gru" in text:
        return "- Name the recurrent cell shown in the figure and describe its gates.\n- How is it similar to and simpler than an LSTM?\n- Why might a practitioner choose a GRU in a real sequence-modeling problem?"
    if "wavenet" in text:
        return "- Explain the architecture shown in the figure.\n- How do causal and dilated convolutions allow the model to handle sequences?\n- Compare this approach with using a recurrent network for long time dependencies."
    return "- Identify the concept shown in the figure.\n- Explain the main components and how information flows through them.\n- Describe why this concept matters for deep learning and give one practical use case."


def render_crop(pdf_path: Path, page: int, crop: tuple[float, float, float, float], out_path: Path) -> None:
    doc = fitz.open(pdf_path)
    pdf_page = doc[page - 1]
    rect = fitz.Rect(*crop) & pdf_page.rect
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=rect, alpha=False)
    pix.save(out_path)


def image_bbox_crop(pdf_path: Path, page: int, large_image_index: int) -> tuple[float, float, float, float]:
    doc = fitz.open(pdf_path)
    blocks = doc[page - 1].get_text("dict")["blocks"]
    images = [
        b
        for b in blocks
        if b["type"] == 1 and (b["bbox"][2] - b["bbox"][0]) > 100 and (b["bbox"][3] - b["bbox"][1]) > 60
    ]
    bbox = images[large_image_index]["bbox"]
    pad = 8
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def source_order(source: str) -> int:
    order = {
        "Sample Exam 2024-06-27": 0,
        "Sample Resit 2024-07-24": 1,
        "Chapter 10": 2,
        "Chapter 14": 3,
        "Chapter 15": 4,
    }
    return order[source]


def load_strong_answers() -> dict[str, str]:
    if not STRONG_ANSWERS_PATH.exists():
        return {}
    raw = json.loads(STRONG_ANSWERS_PATH.read_text(encoding="utf-8"))
    return {item["key"]: item["answer"] for item in raw}


def build_specs() -> list[FigureSpec]:
    specs: list[FigureSpec] = []

    ch10 = [
        (4, "Figure 10-1", "Biological neuron", 0),
        (5, "Figure 10-2", "Multiple layers in a biological neural network (human cortex)", 0),
        (5, "Figure 10-3", "ANNs performing simple logical computations", 1),
        (6, "Figure 10-4", "Threshold logic unit: an artificial neuron which computes a weighted sum", 0),
        (8, "Figure 10-5", "Architecture of a Perceptron with two input neurons, one bias neuron, and three output neurons", 0),
        (10, "Figure 10-6", "XOR classification problem and an MLP that solves it", 0),
        (11, "Figure 10-7", "Architecture of a Multilayer Perceptron with two inputs, one hidden layer of four neurons, and three output neurons", 0),
        (14, "Figure 10-8", "Activation functions and their derivatives", 0),
        (16, "Figure 10-9", "A modern MLP (including ReLU and softmax) for classification", 0),
    ]
    for page, fig_id, caption, idx in ch10:
        specs.append(FigureSpec("Chapter 10", "readings/Chapter 10.pdf", page, fig_id, caption, make_question(caption), topic_answer(caption), image_index=idx))

    ch14 = [
        (3, "Figure 14-1", "Biological neurons in the visual cortex respond to specific patterns in small receptive fields", 0),
        (4, "Figure 14-2", "CNN layers with rectangular local receptive fields", 0),
        (5, "Figure 14-3", "Connections between layers and zero padding", 0),
        (6, "Figure 14-4", "Reducing dimensionality using a stride of 2", 0),
        (7, "Figure 14-5", "Applying two different filters to get two feature maps", 0),
        (8, "Figure 14-6", "Convolutional layers with multiple feature maps, and images with three color channels", 0),
        (11, "Figure 14-7", '"SAME" or "VALID" padding (with input width 13, filter width 6, stride 5)', 0),
        (13, "Figure 14-8", "Max pooling layer (2 x 2 pooling kernel, stride 2, no padding)", 0),
        (14, "Figure 14-9", "Invariance to small translations", 0),
        (15, "Figure 14-10", "Depthwise max pooling can help the CNN learn any invariance", 0),
        (17, "Figure 14-11", "Typical CNN architecture", 0),
        (21, "Figure 14-12", "Generating new training instances from existing ones", 0),
        (23, "Figure 14-13", "Inception module", 0),
        (25, "Figure 14-14", "GoogLeNet architecture", 0),
        (27, "Figure 14-15", "Residual learning", 0),
        (28, "Figure 14-16", "Regular deep neural network and deep residual network", 0),
        (29, "Figure 14-17", "ResNet architecture", 0),
        (29, "Figure 14-18", "Skip connection when changing feature map size and depth", 1),
        (31, "Figure 14-19", "Depthwise separable convolutional layer", 0),
        (32, "Figure 14-20", "SE-Inception module and SE-ResNet unit", 0),
        (33, "Figure 14-21", "An SE block performs feature map recalibration", 0),
        (33, "Figure 14-22", "SE block architecture", 1),
        (41, "Figure 14-23", "Intersection over Union (IoU) metric for bounding boxes", 0),
        (42, "Figure 14-24", "Detecting multiple objects by sliding a CNN across the image", 0),
        (45, "Figure 14-25", "The same fully convolutional network processing a small image and a large image", 0),
        (49, "Figure 14-26", "Semantic segmentation", 0),
        (49, "Figure 14-27", "Upsampling using a transposed convolutional layer", 1),
        (51, "Figure 14-28", "Skip layers recover some spatial resolution from lower layers", 0),
    ]
    for page, fig_id, caption, idx in ch14:
        specs.append(FigureSpec("Chapter 14", "readings/Chapter 14.pdf", page, fig_id, caption, make_question(caption), topic_answer(caption), image_index=idx))

    ch15 = [
        (2, "Figure 15-1", "A recurrent neuron (left) unrolled through time (right)", (120, 360, 500, 525)),
        (3, "Figure 15-2", "A layer of recurrent neurons (left) unrolled through time (right)", (100, 120, 510, 258)),
        (5, "Figure 15-3", "A cell's hidden state and its output may be different", (120, 120, 500, 268)),
        (6, "Figure 15-4", "Seq-to-seq, seq-to-vector, vector-to-seq, and encoder-decoder RNNs", (45, 120, 565, 392)),
        (7, "Figure 15-5", "Backpropagation through time", (105, 155, 520, 333)),
        (8, "Figure 15-6", "Time series forecasting", (80, 55, 535, 230)),
        (11, "Figure 15-7", "Deep RNN (left) unrolled through time (right)", (70, 45, 540, 295)),
        (13, "Figure 15-8", "Forecasting 10 steps ahead, 1 step at a time", (95, 65, 525, 305)),
        (20, "Figure 15-9", "LSTM cell", (85, 55, 540, 360)),
        (23, "Figure 15-10", "GRU cell", (85, 65, 540, 385)),
        (26, "Figure 15-11", "WaveNet architecture", (35, 45, 575, 285)),
    ]
    for page, fig_id, caption, crop in ch15:
        specs.append(FigureSpec("Chapter 15", "readings/Chapter 15.pdf", page, fig_id, caption, make_question(caption), topic_answer(caption), crop=crop))

    sample_specs = [
        (
            "Sample Exam 2024-06-27",
            "sample_tests/20240627_mba_bd_deep_learning_exam.pdf",
            2,
            "Exam Q1",
            "Traditional machine-learning pipeline versus deep-learning pipeline",
            "- Use the figure below to illustrate conceptual differences between traditional classification pipeline and the one based on deep learning.\n- In the context of deep learning, what is end-to-end learning?\n- What are the advantages of end-to-end learning compared to traditional paradigms?\n- Which types of data would you expect deep learning to be effective on and which types of data are more suitable for traditional machine learning approaches?",
            0,
        ),
        (
            "Sample Exam 2024-06-27",
            "sample_tests/20240627_mba_bd_deep_learning_exam.pdf",
            3,
            "Exam Q2",
            "Two convolutional filters producing two feature maps",
            "Figure below illustrates the application of two different filters to produce two feature maps.\n- Which type of Convolutional Neural Networks (CNN) layer are these operations associated with?\n- What is the purpose of filters depicted in the figure?\n- How are the weights of such filters typically learned and why is that a better strategy than using e.g. fixed weights?\n- How do such building blocks contribute to the entire CNN architectures?",
            0,
        ),
        (
            "Sample Exam 2024-06-27",
            "sample_tests/20240627_mba_bd_deep_learning_exam.pdf",
            4,
            "Exam Q3",
            "CNN layer with receptive field and stride",
            "- Name and briefly describe the type of CNN layer illustrated in the figure below.\n- What is the purpose of such layers and how do they fit into overall CNN architecture?",
            0,
        ),
        (
            "Sample Exam 2024-06-27",
            "sample_tests/20240627_mba_bd_deep_learning_exam.pdf",
            5,
            "Exam Q4",
            "RNN sequence architectures",
            "Figure below illustrates seq-to-seq (top left), seq-to-vector (top right), vector-to-seq (bottom left), and Encoder-Decoder (bottom right) Recurrent Neural Networks (RNNs).\n- Briefly describe the type of application each of the illustrated architectures can be used for.\n- Name and briefly describe a major challenge/issue associated with RNNs. How can it be mitigated?",
            0,
        ),
        (
            "Sample Exam 2024-06-27",
            "sample_tests/20240627_mba_bd_deep_learning_exam.pdf",
            6,
            "Exam Q5",
            "Node embedding in geometric deep learning on graphs",
            "- Name and briefly explain two common tasks in geometric deep learning on graphs.\n- Based on the illustration below, briefly explain the motivation and process of node embedding.",
            0,
        ),
        (
            "Sample Exam 2024-06-27",
            "sample_tests/20240627_mba_bd_deep_learning_exam.pdf",
            7,
            "Exam Q6",
            "CLIP-style vision-language model pipeline",
            "- What are the Vision Language Models (VLMs) and how do they work? If it helps, refer to the illustration of CLIP pipeline below.\n- Name several VLM use cases and briefly describe one of them.",
            0,
        ),
        (
            "Sample Resit 2024-07-24",
            "sample_tests/20240724_mba_bd_deep_learning_resit.pdf",
            2,
            "Resit Q1",
            "Feature transformation, representation learning, and latent manifolds",
            "The figures below illustrate the concepts of feature transformation, representation learning and learning of latent manifolds in deep learning. Having that in mind, please answer the following questions:\n- What makes deep learning different and more effective than the traditional ML pipelines?\n- What are the advantages and disadvantages of deep learning?",
            None,
        ),
        (
            "Sample Resit 2024-07-24",
            "sample_tests/20240724_mba_bd_deep_learning_resit.pdf",
            3,
            "Resit Q2",
            "Typical CNN architecture",
            "Figure below illustrates a typical CNN architecture.\n- Briefly explain the structure and role of different types of CNN layers depicted in the figure.\n- On the example of figure below, briefly explain the concept of end-to-end learning.",
            0,
        ),
        (
            "Sample Resit 2024-07-24",
            "sample_tests/20240724_mba_bd_deep_learning_resit.pdf",
            4,
            "Resit Q3",
            "Computer vision tasks",
            "- What makes deep computer vision particularly challenging and different from e.g. classic machine learning with tabular data?\n- Using one of the computer vision tasks from figure below as an example, briefly explain how convolutional neural networks can be effectively applied to solve it.",
            0,
        ),
        (
            "Sample Resit 2024-07-24",
            "sample_tests/20240724_mba_bd_deep_learning_resit.pdf",
            5,
            "Resit Q4",
            "Recurrent neurons unrolled through time",
            "Figure below illustrates a layer of recurrent neurons (left) unrolled through time (right).\n- What makes recurrent neurons and Recurrent Neural Networks (RNNs) particularly effective in processing sequences?\n- What is their main limitation and how Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) help address it?",
            0,
        ),
        (
            "Sample Resit 2024-07-24",
            "sample_tests/20240724_mba_bd_deep_learning_resit.pdf",
            6,
            "Resit Q5",
            "Graph learning tasks",
            "- Name and briefly describe machine/deep learning tasks shown in the figure below.\n- What makes geometric deep learning on graphs challenging and why conventional CNNs and RNNs are not directly applicable to graphs/networks?\n- (a) Name the task. (b) Name the task.",
            None,
        ),
    ]
    for source, pdf, page, fig_id, caption, question, idx in sample_specs:
        crop = None
        if idx is None:
            doc = fitz.open(ROOT / pdf)
            images = [b for b in doc[page - 1].get_text("dict")["blocks"] if b["type"] == 1]
            xs = [v for b in images for v in (b["bbox"][0], b["bbox"][2])]
            ys = [v for b in images for v in (b["bbox"][1], b["bbox"][3])]
            crop = (min(xs) - 8, min(ys) - 8, max(xs) + 8, max(ys) + 8)
        specs.append(
            FigureSpec(
                source,
                pdf,
                page,
                fig_id,
                caption,
                question,
                topic_answer(caption),
                image_index=idx,
                crop=crop,
            )
        )

    return sorted(specs, key=lambda spec: source_order(spec.source))


def build_html_document(items: list[dict[str, str]]) -> str:
    pages: list[str] = []
    toc_links: list[str] = []
    for i, item in enumerate(items, 1):
        page_id = f"item-{i:03d}"
        source = html.escape(item["source"])
        figure_id = html.escape(item["figure_id"])
        caption = html.escape(item["caption"])
        image = html.escape(item["image"])
        question = html.escape(item["question"])
        answer = html.escape(item["answer"])
        alt = html.escape(f"{item['figure_id']}: {item['caption']}")
        toc_links.append(
            f'<a class="toc-link" href="#{page_id}-question" data-source="{source}">{i:02d}. {figure_id}</a>'
        )
        pages.append(
            f"""
    <section id="{page_id}-question" class="page page-question" data-source="{source}" data-index="{i}" data-total="{len(items)}">
      <div class="page-inner">
        <div class="meta"><span class="badge">Question</span><span>{i:02d} / {len(items):02d}</span><span>{source}</span></div>
        <h2>{figure_id}</h2>
        <p class="caption">{caption}</p>
        <figure class="figure-wrap figure-large">
          <img src="{image}" alt="{alt}" loading="lazy">
        </figure>
        <div class="prompt">
          <h3>Study prompt</h3>
          <p class="question">{question}</p>
        </div>
        <p class="hint"><a href="#{page_id}-answer">View answer ↓</a> · scroll down · <kbd>↓</kbd></p>
      </div>
    </section>
    <section id="{page_id}-answer" class="page page-answer" data-source="{source}" data-index="{i}" data-total="{len(items)}">
      <div class="page-inner">
        <div class="meta"><span class="badge badge-answer">Answer</span><span>{i:02d} / {len(items):02d}</span><span>{source}</span></div>
        <h2>{figure_id}</h2>
        <p class="caption">{caption}</p>
        <figure class="figure-wrap figure-small">
          <img src="{image}" alt="{alt}" loading="lazy">
        </figure>
        <div class="answer-panel">
          <h3>Strong answer</h3>
          <p class="answer">{answer}</p>
        </div>
        <p class="hint"><a href="#{page_id}-question">Back to question ↑</a> · <kbd>↑</kbd></p>
      </div>
    </section>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deep Learning Figures Quiz</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18202a;
      --muted: #5d6673;
      --line: #d8dde5;
      --paper: #ffffff;
      --bg: #f4f6f8;
      --accent: #0f6b57;
    }}
    * {{ box-sizing: border-box; }}
    html {{
      scroll-behavior: smooth;
      scroll-snap-type: y mandatory;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .top-bar {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding: 10px clamp(12px, 3vw, 28px);
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    .top-bar a.home {{
      font-weight: 600;
      color: var(--accent);
      text-decoration: none;
      margin-right: 4px;
    }}
    .progress {{
      margin-left: auto;
      color: var(--muted);
      font-size: .9rem;
      white-space: nowrap;
    }}
    .filter, .nav-btn, .toc-toggle {{
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 6px 10px;
      font: inherit;
      cursor: pointer;
    }}
    .filter.active, .nav-btn:hover, .toc-toggle:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .intro {{
      padding: 88px clamp(18px, 4vw, 48px) 28px;
      background: var(--paper);
      border-bottom: 1px solid var(--line);
      scroll-snap-align: start;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.6rem, 3vw, 2.4rem);
    }}
    .summary {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.5;
      max-width: 900px;
    }}
    .toc {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      max-height: 0;
      overflow: hidden;
      transition: max-height .25s ease;
    }}
    .toc.open {{
      max-height: 240px;
      overflow-y: auto;
      margin-top: 12px;
    }}
    .toc-link {{
      display: inline-block;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      text-decoration: none;
      color: var(--ink);
      font-size: .85rem;
      background: var(--bg);
    }}
    .toc-link:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .slides {{
      padding-top: 56px;
    }}
    .page {{
      min-height: 100vh;
      min-height: 100dvh;
      scroll-snap-align: start;
      scroll-snap-stop: always;
      display: flex;
      align-items: stretch;
      border-bottom: 1px solid var(--line);
      background: var(--paper);
    }}
    .page-answer {{
      background: #f8fbfa;
    }}
    .page-inner {{
      width: min(1100px, 100%);
      margin: 0 auto;
      padding: clamp(20px, 4vw, 40px) clamp(18px, 4vw, 48px) 32px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      flex: 1;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: .82rem;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .badge {{
      background: #e9edf1;
      padding: 4px 8px;
      border-radius: 4px;
      color: var(--ink);
    }}
    .badge-answer {{
      background: #dff3eb;
      color: var(--accent);
    }}
    h2 {{
      margin: 0;
      font-size: clamp(1.2rem, 2.5vw, 1.6rem);
    }}
    h3 {{
      margin: 0 0 6px;
      font-size: 1rem;
      color: var(--accent);
    }}
    .caption, .question, .answer, .hint {{
      margin: 0;
      line-height: 1.5;
    }}
    .caption {{ color: var(--muted); }}
    .question, .answer {{ white-space: pre-line; }}
    .figure-wrap {{
      margin: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .figure-large img {{
      width: 100%;
      max-height: min(52vh, 520px);
      object-fit: contain;
    }}
    .figure-small img {{
      width: 100%;
      max-height: min(28vh, 240px);
      object-fit: contain;
      opacity: .95;
    }}
    .prompt, .answer-panel {{
      flex: 1;
      min-height: 0;
      overflow: auto;
    }}
    .answer-panel {{
      background: #eef7f3;
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      padding: 16px 18px;
    }}
    .hint {{
      color: var(--muted);
      font-size: .9rem;
    }}
    .hint a {{
      color: var(--accent);
    }}
    kbd {{
      font: inherit;
      padding: 2px 6px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--bg);
    }}
    .hidden {{ display: none !important; }}
    @media (max-width: 720px) {{
      .filter {{ display: none; }}
      .top-bar .filters {{ display: none; }}
    }}
  </style>
</head>
<body>
  <nav class="top-bar" aria-label="Quiz navigation">
    <a class="home" href="#top">Quiz</a>
    <span class="filters">
      <button class="filter active" type="button" data-filter="all">All</button>
      <button class="filter" type="button" data-filter="Chapter 10">Ch.10</button>
      <button class="filter" type="button" data-filter="Chapter 14">Ch.14</button>
      <button class="filter" type="button" data-filter="Chapter 15">Ch.15</button>
      <button class="filter" type="button" data-filter="Sample Exam 2024-06-27">Exam</button>
      <button class="filter" type="button" data-filter="Sample Resit 2024-07-24">Resit</button>
    </span>
    <button class="nav-btn" type="button" id="prev-btn" aria-label="Previous page">←</button>
    <button class="nav-btn" type="button" id="next-btn" aria-label="Next page">→</button>
    <button class="toc-toggle" type="button" id="toc-toggle">Index</button>
    <span class="progress" id="progress">1 / {len(items) * 2} pages</span>
  </nav>

  <header class="intro" id="top">
    <h1>Deep Learning Figures Quiz</h1>
    <p class="summary">Full-screen study deck: one page per question, one page per answer. Use arrow keys, scroll, or the nav buttons. {len(items)} figures from chapters and sample exams.</p>
    <nav class="toc" id="toc" aria-label="Figure index">
      {''.join(toc_links)}
    </nav>
  </header>

  <main class="slides">
    {''.join(pages)}
  </main>

  <script>
    const pages = Array.from(document.querySelectorAll('.page'));
    const progressEl = document.getElementById('progress');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const toc = document.getElementById('toc');
    const tocToggle = document.getElementById('toc-toggle');

    function visiblePages() {{
      return pages.filter((page) => !page.classList.contains('hidden'));
    }}

    function pageLabel(page) {{
      const kind = page.classList.contains('page-answer') ? 'Answer' : 'Question';
      return `${{kind}} · ${{page.dataset.index}} / ${{page.dataset.total}}`;
    }}

    function updateProgress() {{
      const visible = visiblePages();
      const idx = visible.findIndex((page) => {{
        const rect = page.getBoundingClientRect();
        return rect.top >= -80 && rect.top < window.innerHeight * 0.45;
      }});
      const current = idx >= 0 ? visible[idx] : visible[0];
      if (current) {{
        progressEl.textContent = `${{pageLabel(current)}} · ${{visible.indexOf(current) + 1}} / ${{visible.length}} pages`;
      }}
    }}

    function scrollToPage(page, behavior = 'smooth') {{
      if (!page) return;
      page.scrollIntoView({{ behavior, block: 'start' }});
    }}

    function currentPageIndex() {{
      const visible = visiblePages();
      const idx = visible.findIndex((page) => {{
        const rect = page.getBoundingClientRect();
        return rect.top >= -80 && rect.top < window.innerHeight * 0.45;
      }});
      return idx >= 0 ? idx : 0;
    }}

    prevBtn.addEventListener('click', () => {{
      const visible = visiblePages();
      const idx = currentPageIndex();
      scrollToPage(visible[Math.max(0, idx - 1)]);
    }});

    nextBtn.addEventListener('click', () => {{
      const visible = visiblePages();
      const idx = currentPageIndex();
      scrollToPage(visible[Math.min(visible.length - 1, idx + 1)]);
    }});

    tocToggle.addEventListener('click', () => toc.classList.toggle('open'));

    document.querySelectorAll('.filter').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.filter').forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        pages.forEach((page) => {{
          page.classList.toggle('hidden', filter !== 'all' && page.dataset.source !== filter);
        }});
        document.querySelectorAll('.toc-link').forEach((link) => {{
          link.classList.toggle('hidden', filter !== 'all' && link.dataset.source !== filter);
        }});
        updateProgress();
      }});
    }});

    document.addEventListener('keydown', (event) => {{
      const visible = visiblePages();
      const idx = currentPageIndex();
      if (['ArrowDown', 'PageDown', 'j'].includes(event.key)) {{
        event.preventDefault();
        scrollToPage(visible[Math.min(visible.length - 1, idx + 1)]);
      }}
      if (['ArrowUp', 'PageUp', 'k'].includes(event.key)) {{
        event.preventDefault();
        scrollToPage(visible[Math.max(0, idx - 1)]);
      }}
    }});

    window.addEventListener('scroll', updateProgress, {{ passive: true }});
    updateProgress();
  </script>
</body>
</html>
"""


def write_html(items: list[dict[str, str]]) -> None:
    html_doc = build_html_document(items)
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_HTML_PATH.write_text(html_doc, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    strong_answers = load_strong_answers()
    items: list[dict[str, str]] = []
    for spec in build_specs():
        pdf_path = ROOT / spec.pdf
        filename = f"{slug(spec.source)}-{slug(spec.figure_id)}.png"
        out_path = ASSET_DIR / filename
        crop = spec.crop
        if crop is None:
            if spec.image_index is None:
                raise ValueError(f"Missing crop or image index for {spec.figure_id}")
            crop = image_bbox_crop(pdf_path, spec.page, spec.image_index)
        render_crop(pdf_path, spec.page, crop, out_path)
        shutil.copy2(out_path, DOCS_ASSET_DIR / filename)
        items.append(
            {
                "source": spec.source,
                "figure_id": spec.figure_id,
                "caption": spec.caption,
                "question": spec.question,
                "answer": strong_answers.get(f"{spec.source}|{spec.figure_id}", spec.answer),
                "image": f"assets/{filename}",
            }
        )

    write_html(items)
    (OUT_DIR / "manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} figures ({len(items) * 2} full pages) to:")
    print(f"  {HTML_PATH}")
    print(f"  {DOCS_HTML_PATH}")


if __name__ == "__main__":
    main()

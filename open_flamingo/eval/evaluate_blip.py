import argparse
import json
import os
import random
import uuid
from collections import defaultdict
from typing import Callable
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

import more_itertools
import numpy as np
import torch
from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import COCOFlickrDataset, VQAv2Dataset, ImageNetDataset
from tqdm import tqdm
from vqa_metric import compute_vqa_accuracy
from open_flamingo.eval.classification import compute_per_sample_probs, \
    compute_per_sample_loss
from open_flamingo.eval.imagenet_utils import openai_imagenet_classnames, \
    IMAGENET_1K_CLASS_ID_TO_LABEL

parser = argparse.ArgumentParser()
parser.add_argument("--blip_path", default="Salesforce/blip2-opt-2.7b", type=str)
# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=2,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0, 1, 2],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000,
    help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument("--eval_coco", action="store_true", default=False,
                    help="Whether to evaluate on COCO.")

parser.add_argument("--eval_vqav2", action="store_true", default=False,
                    help="Whether to evaluate on VQAV2.")
parser.add_argument("--eval_imagenet", action="store_true", default=False,
                    help="Whether to evaluate on ImageNet.")

parser.add_argument("--eval_flickr30", action="store_true", default=False,
                    help="Whether to evaluate on Flickr30.")

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default=None,
)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    default="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    default="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
)

## Imagenet dataset
parser.add_argument("--imagenet_root",
                    type=str,
                    default="/tmp")


def main():
    args = parser.parse_args()

    # load model
    processor = AutoProcessor.from_pretrained(args.blip_path)
    model = Blip2ForConditionalGeneration.from_pretrained(args.blip_path, torch_dtype=torch.float16)
    
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
    
    model.to(args.device if args.device >= 0 else "cpu")

    results = defaultdict(list)

    if args.eval_flickr30:
        print("Evaluating on Flickr30...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.flickr_image_dir_path,
                    annotations_json_path=args.flickr_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    is_flickr=True
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["flickr30"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})
    results = defaultdict(list)

    if args.eval_coco:

        print("Evaluating on COCO...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.coco_image_dir_path,
                    annotations_json_path=args.coco_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["coco"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    if args.eval_vqav2:

        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            results["vqav2"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    if args.eval_imagenet:

        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_imagenet(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    imagenet_root=args.imagenet_root
                )
                print(f"Shots {shot} Trial {trial} "
                      f"ImageNet score: {imagenet_score}")
                scores.append(imagenet_score)
            print(f"Shots {shot} Mean ImageNet score: {np.mean(scores)}")
            results["imagenet"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices,
                                     query_set_size):
    # get in context samples
    in_context_samples = [full_dataset[i]
                          for i in random_indices[:query_set_size]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:])
    return in_context_samples, eval_dataset


def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = image_processor(
            images=[s["image"] for s in in_context_samples],
            return_tensors="pt",
        )["pixel_values"]
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


def get_context_text(get_prompt: Callable[[dict], str], in_context_samples,
                     effective_num_shots, num_shots) -> str:
    context_text = (
        "".join([get_prompt(s) for s in in_context_samples]
                ) if effective_num_shots > 0 else ""
    )

    if num_shots == 0:
        context_text = context_text.replace("<image>", "")
    return context_text


def prepare_batch_images(batch, image_processor, context_images,
                         num_shots):
    batch_images = None
    for b, sample_imgs in zip(batch, context_images):
        b_image = image_processor(images=[b["image"]], return_tensors="pt")[
            "pixel_values"
        ]
        b_image = b_image.unsqueeze(1).unsqueeze(0)
        b_image = (
            torch.cat([sample_imgs, b_image], dim=1)
            if num_shots > 0
            else b_image
        )

        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images

def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [ random.sample(query_set, num_samples) for _ in range(batch_size) ]
    
def get_outputs(model, batch_images, device, attention_mask,
                max_generation_length, num_beams, length_penalty, input_ids):
    with torch.inference_mode():
        outputs = model.generate(
            batch_images.to(device if device >= 0 else "cpu"),
            input_ids.to(device if device >= 0 else "cpu"),
            attention_mask=attention_mask.to(
                device if device >= 0 else "cpu"),
            max_new_tokens=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    outputs = outputs[:, len(input_ids[0]):]
    return outputs


def evaluate_coco_flickr(
        model,
        tokenizer,
        image_processor,
        batch_size,
        image_dir_path,
        annotations_json_path,
        seed=42,
        max_generation_length=10,
        num_beams=3,
        length_penalty=-2.0,
        num_samples=5000,
        query_set_size=2048,
        num_shots=8,
        device=-1,
        is_flickr=False,
):
    """Evaluate a model on COCO dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        image_dir_path (str, optional): path to the directory containing the images.
        annotations_json_path (str, optional): path to the json file containing the annotations.
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 10.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000.
        query_set_size (int, optional): number of samples to use for query set. Defaults to 2048.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1.
        num_workers (int, optional): number of workers to use for dataloader. Defaults to 4.
        is_flickr (bool): defines if that data is COCO or Flickr. Defaults to False (COCO).

    Returns:
        float: CIDEr score

    """

    full_dataset = COCOFlickrDataset(
        image_dir_path=image_dir_path, annotations_path=annotations_json_path,
        is_flickr=is_flickr,
    )
    random_indices = get_random_indices(num_samples, query_set_size,
                                        full_dataset, seed)

    _, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices, query_set_size=query_set_size)

    model.eval()

    predictions = defaultdict()

    desc = 'Running inference Flickr30' if is_flickr else 'Running inference COCO'

    for batch in more_itertools.chunked(
            tqdm(eval_dataset, desc=desc), batch_size
    ):  
        batch_images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt"
        )["pixel_values"]
        
        batch_text = [f"A photo of" for _ in range(len(batch))]
                
        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        input_ids = encodings["input_ids"].to(device if device >= 0 else "cpu")
        attention_mask = encodings["attention_mask"].to(
            device if device >= 0 else "cpu"
        )
        
        batch_images = batch_images.to(device if device >= 0 else "cpu", torch.float16)

        outputs = model.generate(pixel_values=batch_images, attention_mask=attention_mask, input_ids=input_ids, max_new_tokens=max_generation_length, num_beams=num_beams, length_penalty=length_penalty)
        
        new_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # NOTE: Not sure why I get partial EOC tokens, but this fixes it
        new_predictions = [p.replace("\n", "").split("<")[0] for p in new_predictions]

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "")
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = {
                "caption": new_predictions[i],
            }

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    results_path = f"flickrresults_{random_uuid}.json" if is_flickr \
        else f"cocoresults_{random_uuid}.json"
    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": predictions[k]["caption"]}
                    for k in predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=annotations_json_path,
    )

    # delete the temporary file
    os.remove(results_path)

    return metrics["CIDEr"] * 100.0


def evaluate_vqa(
        model,
        tokenizer,
        image_processor,
        batch_size,
        image_dir_path,
        questions_json_path,
        annotations_json_path,
        seed=42,
        max_generation_length=5,
        num_beams=3,
        length_penalty=-2.0,
        num_samples=5000,
        query_set_size=2048,
        num_shots=8,
        device=-1,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        image_dir_path (str): path to image directory
        questions_json_path (str): path to questions json file
        annotations_json_path (str): path to annotations json file
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        query_set_size (int, optional): size of the query set. Defaults to 2048.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        num_workers (int, optional): number of workers to use. Defaults to 4.

    Returns:
        float: accuracy score
    """
    full_dataset = VQAv2Dataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
    )

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, query_set_size,
                                        full_dataset, seed)

    def get_prompt(sample, train=True):
        return f"Question:{sample['question'].strip()} Answer:{sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"

    _, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices,
        query_set_size=query_set_size)

    model.eval()
    predictions = []

    for batch in more_itertools.chunked(
            tqdm(eval_dataset, desc="Running inference"), batch_size
    ):  
                
        batch_images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt"
        )["pixel_values"]
        
        batch_text = [get_prompt(s, train=False) for s in batch]
        
        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encodings["input_ids"].to(device if device >= 0 else "cpu")
        attention_mask = encodings["attention_mask"].to(
            device if device >= 0 else "cpu"
        )
        
        batch_images = batch_images.to(device if device >= 0 else "cpu", torch.float16)

        outputs = model.generate(pixel_values=batch_images, attention_mask=attention_mask, input_ids=input_ids, max_new_tokens=max_generation_length, num_beams=num_beams, length_penalty=length_penalty)
        
        new_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        new_predictions = [p.replace("\n", "").split("<")[0] for p in new_predictions]
                
        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"vqaresults_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"vqaresults_{random_uuid}.json", questions_json_path,
        annotations_json_path
    )

    # delete the temporary file
    os.remove(f"vqaresults_{random_uuid}.json")

    return acc


def evaluate_imagenet(
        model,
        tokenizer,
        image_processor,
        batch_size: int,
        imagenet_root: str,
        seed: int = 42,
        num_samples: int = 5000,
        num_shots: int = 8,
        device: int = -1,
):
    """
    Evaluate a model on ImageNet dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).

    Returns:
        float: accuracy score
    """

    full_dataset = ImageNetDataset(root=imagenet_root)

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, effective_num_shots,
                                        full_dataset, seed)

    eoc_token = "<|endofchunk|>"

    def _imagenet_prompt(class_name, is_context: bool = True):
        """Construct an imagenet prompt for a given label."""
        prefix = "<image>A photo of a "
        if is_context:
            return prefix + class_name.strip()
        else:
            # Not a context example; insert EOS token before the class name
            # so that we can compute the loss on the class name tokens only.
            return prefix + tokenizer.eos_token + class_name.strip()

    def get_imagenet_prompt(x: dict, is_context: bool = True) -> str:
        """Construct an ImageNet prompt for an example, using its label."""
        return _imagenet_prompt(x['class_name'], is_context=is_context)

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices,
        effective_num_shots=effective_num_shots)

    model.eval()
    # Predictions based on the class target sequence with the maximal predicted probability
    predictions_max_prob = []
    # Predictions based on the class target sequence with the minimal loss on the model logits
    predictions_min_loss = []
    labels = []

    context_images = get_context_images(image_processor=image_processor,
                                        in_context_samples=in_context_samples,
                                        num_shots=num_shots)

    context_text = get_context_text(get_imagenet_prompt,
                                    in_context_samples=in_context_samples,
                                    effective_num_shots=effective_num_shots,
                                    num_shots=num_shots)

    for i, batch in enumerate(more_itertools.chunked(eval_dataset, batch_size)):
        print(f"processing batch {i} of {len(eval_dataset)}")
        batch_per_class_probs = []
        batch_per_class_losses = []
        batch_images = prepare_batch_images(batch=batch,
                                            image_processor=image_processor,
                                            context_images=context_images,
                                            num_shots=num_shots)

        # For each ImageNet class, construct the output prompt, compute its
        # completion 'loss'. The class with the lowest completion loss would
        # be the predicted label.
        for imagenet_class_name in tqdm(openai_imagenet_classnames):
            batch_text = [context_text
                          + _imagenet_prompt(imagenet_class_name, False)
                          + eoc_token] * batch_size

            tokenizer.padding_side = "left"
            encodings = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            device = device if device >= 0 else "cpu"

            # input_ids has shape [batch_size, seq_len]
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            batch_images = batch_images.to(device)

            outputs = model(batch_images, input_ids, attention_mask)

            per_sample_probs = compute_per_sample_probs(encodings=encodings,
                                                        tokenizer=tokenizer,
                                                        outputs=outputs)
            per_sample_loss = compute_per_sample_loss(encodings=encodings,
                                                      tokenizer=tokenizer,
                                                      outputs=outputs)
            batch_per_class_probs.append(per_sample_probs.detach())
            batch_per_class_losses.append(per_sample_loss.detach())

        # Tensor of shape [batch_size, 1000] where the [i,j]th element is
        # the (probability or loss) for batch element i on imagenet class j.
        batch_probs = torch.stack(batch_per_class_probs, 1)
        batch_losses = torch.stack(batch_per_class_losses, 1)

        predictions_max_prob.extend(
            torch.argmax(batch_probs, 1).detach().tolist())
        predictions_min_loss.extend(
            torch.argmin(batch_losses, 1).detach().tolist())
        labels.extend(x['class_id'] for x in batch)

    acc_max_prob = (np.array(predictions_max_prob) == np.array(labels)).mean()
    acc_min_loss = (np.array(predictions_min_loss) == np.array(labels)).mean()
    print(f"[DEBUG] ImageNet accuracy with max prob method is {acc_max_prob}")
    print(f"[DEBUG] ImageNet accuracy with min loss method is {acc_min_loss}")
    print(f"[DEBUG] printing ImageNet predictions and labels:")
    for yhat_prob, yhat_loss, y in zip(predictions_max_prob,
                                       predictions_min_loss,
                                       labels):
        print(" " * 30 + f"label: {IMAGENET_1K_CLASS_ID_TO_LABEL[y]}"
                         f"\nprediction (max prob method): "
                         f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_prob]}"
                         f"\nprediction (min loss method): "
                         f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_loss]}\n"
                         "#" * 25)
    return acc_max_prob


if __name__ == "__main__":
    main()
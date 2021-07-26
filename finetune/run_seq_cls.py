import argparse
import json
import logging
import os
import glob
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
from tqdm.auto import tqdm
from time import sleep

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    init_logger,
    set_seed,
    compute_metrics
)
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from processor import seq_cls_processors as processors
from processor import seq_cls_output_modes as output_modes

logger = logging.getLogger(__name__)


def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_text = pd.read_csv('/content/KoELECTRA/finetune/data/SCIC/SCIC_train.txt', sep='\t')
    print(train_text['Review'])
    if args.max_steps > 0:
        t_total = args.max_steps
        print()
        print(t_total, len(train_dataloader), args.gradient_accumulation_steps)
        print()
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs * 2
        print('max steps: ' + str(t_total), 'length of train data: ' + str(len(train_dataloader)),
              args.gradient_accumulation_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    # for epoch in tqdm(range(int(args.num_train_epochs))):
    #     sleep(0.1)
    for epoch in tqdm(mb):
        epoch_iterator = progress_bar(train_dataloader, parent=mb)

        # 내가 수정한 부분
        with tqdm(total=t_total / args.num_train_epochs / 2) as pbar:
            #
            for step, batch in enumerate(epoch_iterator):
                #
                # print(len(batch))
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                if args.model_type not in ["distilkobert", "xlm-roberta"]:
                    inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
                outputs = model(**inputs)
                loss = outputs[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                # 내가 추가한 부분
                # logits = batch[0]
                # tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
                #     args.model_name_or_path,
                #     do_lower_case=args.do_lower_case
                # )
                #
                # temp = logits.detach().cpu().numpy()
                # for i in range(len(temp)):
                #     # print(i)
                #     review_list = list(temp[i])
                #     while 0 in review_list:
                #         review_list.remove(0)
                #     del review_list[0]
                #     del review_list[-1]
                #     review_list = np.asarray(review_list)
                #     print(tokenizer.decode(review_list), batch[3][i])
                ##

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        len(train_dataloader) <= args.gradient_accumulation_steps
                        and (step + 1) == len(train_dataloader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # 내가 수정한 부분
                    print("loss: " + str(tr_loss / global_step))
                    sleep(0.1)
                    pbar.update()
                    #

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if args.evaluate_test_during_training:
                            evaluate(args, model, train_text, test_dataset, "test", global_step)
                        else:
                            evaluate(args, model, train_text, dev_dataset, "dev", global_step)

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to {}".format(output_dir))

                        if args.save_optimizer:
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))
                if args.max_steps > 0 and global_step > args.max_steps:
                    break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step, train_text


def evaluate(args, model, train_text, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    out_input_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }

            # 내가 쓴 곳
            # tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
            #     args.model_name_or_path,
            #     do_lower_case=args.do_lower_case
            # )
            # for i in range(len(inputs)):
            #     print(tokenizer.decode(inputs['input_ids'][i]), inputs['labels'])

            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            out_input_ids = inputs['input_ids'].detach().cpu().numpy()
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            out_input_ids = np.append(out_input_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # 내가 수정한 부분

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )

    label_dict = {'None': 0, '상담원': 1, '상담시스템': 2, '고객서비스': 3, '혜택': 4, '할부금융상품': 5,
                  '커뮤니티서비스': 6, '카드이용/결제': 7, '카드상품': 8, '청구입금': 9, '심사/한도': 10,
                  '생활편의서비스': 11, '상담/채널': 12, '리스렌탈상품': 13, '라이프서비스': 14, '금융상품': 15,
                  '고객정보관리': 16, '가맹점매출/승인': 17, '가맹점대금': 18, '가맹점계약': 19, '삼성카드': 20, '기타': 21}
    label_dict = dict((v, k) for k, v in label_dict.items())
    df_review = []
    # temp_review = []
    df_label = np.vectorize(label_dict.get)(out_label_ids)
    df_prediction = np.vectorize(label_dict.get(np.argmax(preds)))
    for i in range(len(out_input_ids)):
        review_list = list(out_input_ids[i])


        # temp_review.append(str(x) for x in out_input_ids[i])


        while 0 in review_list:
            review_list.remove(0)
        del review_list[0]
        del review_list[-1]
        df_review.append(tokenizer.decode(review_list))
        # print(review_list, label_dict[out_label_ids[i] - 1], label_dict[np.argmax(preds[i]) - 1])
    df_data = {'Review': df_review, 'Label': df_label, 'Prediction': df_prediction}
    df = pd.DataFrame(df_data)
    df_train_data = {'Review': train_text['Review'], 'Label': train_text['Label']}
    df_from_train = pd.DataFrame(df_train_data)
    # Dodged Bar Chart (with same X coordinates side by side)

    bar_width = 0.35
    alpha = 0.5
    label_lst = list(label_dict.values())
    index = np.arange(len(label_lst))
    # print(index)
    count_list, acc_list, acc_cnt = [0 for _ in range(len(label_dict))], \
                                    [0 for _ in range(len(label_dict))], [0 for _ in range(len(label_dict))]
    count_labels = df_from_train.groupby('Label', as_index=False).Review.count()
    acc_labels = df[df['Label'] == df['Prediction']].groupby('Label').Review.count()
    # print('hey', df_from_train['Label'][5])
    # print(count_labels)
    viable_label = list(count_labels['Label'])
    for i in range(len(viable_label)):
        count_list[viable_label[i]] = count_labels['Review'][i]
    # for i in range(len(df_from_train['Label'])):
    #     print(count_list[i], count_labels[i])
    #     count_list[int(df_from_train['Label'][i])] = count_labels[i]
    # print('out', len(out_label_ids))
    # print(len(list(out_label_ids)))
    for j in range(len(out_label_ids)):
        if out_label_ids[j] == np.argmax(preds, axis=1)[j]:
            acc_list[out_label_ids[j]] += 1
        acc_cnt[out_label_ids[j]] += 1
    acc_tot = np.divide(acc_list, acc_cnt)
    acc_tot[np.isnan(acc_tot)] = 0
    print(count_list)
    print(acc_tot)
    plt.subplot(2, 1, 1)
    plt.title('Bar Chart of Labels Count and Accuracy', fontsize=15)
    p1 = plt.bar(index, count_list,
                 bar_width,
                 color='b',
                 alpha=alpha,
                 label='Count')
    plt.ylabel('Count of Labels', fontsize=12)
    plt.xticks([], [])
    plt.legend((p1[0],), ('Count',), fontsize=10)
    plt.subplot(2, 1, 2)
    p2 = plt.bar(index + bar_width, acc_tot,
                 bar_width,
                 color='b',
                 alpha=alpha,
                 label='Accuracy')
    plt.ylabel('Accuracy by Labels', fontsize=12)
    plt.xlabel('Label', fontsize=12)
    plt.xticks(index, label_lst, fontsize=10)
    plt.legend((p2[0],), ('Accuracy',), fontsize=10)
    plt.show()
        # for i in range(len(out_label_ids)):
        #     print(tokenizer.decode(out_ids[i]), out_label_ids[i], preds[i])
        # print(type(out_label_ids), type(preds))
        # print(out_label_ids, preds)
        #

    eval_loss = eval_loss / nb_eval_steps
    if output_modes[args.task] == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_modes[args.task] == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task, out_label_ids, preds)

    # numpy_data = np.array(out_label_ids, preds)
    # df = pd.DataFrame(data=numpy_data, index=["row1", "row2"], columns=["column1", "column2"])

    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results


def main(cli_args):
    train_text = None
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = processors[args.task](args)
    labels = processor.get_labels()
    if output_modes[args.task] == "regression":
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task]
        )
    else:
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task],
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss, train_text = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, train_text, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)

from tqdm.auto import tqdm


def move_batch_to_device(batch, device):
    return {
        'indices': batch["indices"],
        'frames_map': batch["frames_map"].to(device),
        'temporal_masks': batch["temporal_masks"].to(device),
        'captioner_attention_masks': batch["captioner_attention_masks"].to(device),
        'I_frames': batch["I_frames"].to(device),
        'motion_vectors': batch["motion_vectors"].to(device),
        'residuals': batch["residuals"].to(device) if batch["residuals"] is not None else None,
        'transcripts_features': batch["transcripts_features"].to(device) if batch["transcripts_features"] is not None else None,
        'spotting_labels': batch["spotting_labels"].to(device) if batch["spotting_labels"] is not None else None,
        'captions': batch["captions"] if batch["captions"] is not None else None,
        'captions_input_ids': batch["captions_input_ids"].to(device) if batch["captions_input_ids"] is not None else None
    }


def train_one_epoch(model, data_loader, short_memory_len, umt5_enabled, exclude_bg, optimizer, scheduler, device, profiler, verbose=0):
    progress_bar = tqdm(enumerate(data_loader), desc=f"Train", total=len(data_loader))
    for i, batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        spotting_logits, caption_logits, step_loss = model(**batch)

        step_loss.backward()
        optimizer.step()
        scheduler.step()
        profiler.step()

def evaluate_one_epoch_sn(model, dataloader, dataset, split, umt5_enabled, base_dir, device, profiler):
    for i, batch in tqdm(enumerate(dataloader), desc=f"Predicting", total=len(dataloader)):
        batch = move_batch_to_device(batch, device)
        spotting_logits, caption_logits, _ = model(**batch, num_beams=1, max_new_tokens=50)
        profiler.step()


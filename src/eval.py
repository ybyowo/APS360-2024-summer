# Choose the best model checkpoint based on highest average precision (AP)
max_index = val_aps.index(max(val_aps))
print(f"Best model at epoch {max_index}")
best_model_path = f"FastRCNN_learning_rate_{lr}_batch_size_{batch_size}_epoch_{max_index}.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))

# Create a dataset and loader for self-taken images and combine with the test set
new_dataset = LicensePlateDataset('lp_loc',  transforms=transform)
new_dataset_loader = DataLoader(new_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
combined_test_set = ConcatDataset([test_dataset, new_dataset])
combined_test_loader = DataLoader(combined_test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Recall, precision, and average precision for the best model on the training set
train_recall, train_precision, train_ap = evaluate_model(model, train_loader, device)
print(f"Training Set - Recall: {train_recall}, Precision: {train_precision}, Average Precision: {train_ap}")

# Recall, precision, and average precision for the best model on the validation set
val_recall, val_precision, val_ap = evaluate_model(model, val_loader, device)
print(f"Validation Set - Recall: {val_recall}, Precision: {val_precision}, Average Precision: {val_ap}")

# Recall, precision, and average precision for the best model on the combined test set
combined_recall, combined_precision, combined_ap = evaluate_model(model, combined_test_loader, device)
print(f"Combined Test Set - Recall: {combined_recall}, Precision: {combined_precision}, Average Precision: {combined_ap}")

# Visualize the bounding box predictions for the combined test set
model.eval()
model = model.to('cpu')
for images, boxes, labels in combined_test_loader:
    with torch.no_grad():
        targets = []
        for box, label in zip(boxes, labels):
            target = {}
            target["boxes"] = box
            target["labels"] = label
            targets.append(target)
        images = images[:,:3,:,:]
        pred = model(images)

        visualize_image_with_boxes(images[0], pred[0])

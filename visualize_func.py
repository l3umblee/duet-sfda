import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def visualize_latent_space(all_feas, all_labels, method, epoch, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=5.0,learning_rate=learning_rate, n_iter=n_iter)
    tsne_results = tsne.fit_transform(all_feas)

    filename = f'./ScatterPlots/tsne_{method}_{epoch}.png'
    if n_components == 2:
        plt.figure(figsize=(10, 7))
        plt.grid(True, linestyle='--', color="white", linewidth=1.0, zorder=0)
        plt.gca().set_facecolor("#EAEAEA")
        plt.xticks(np.linspace(min(tsne_results[:, 0]), max(tsne_results[:, 0]), num=5), [])
        plt.yticks(np.linspace(min(tsne_results[:, 1]), max(tsne_results[:, 1]), num=5), [])
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='tab10', edgecolors='black', linewidth=0.5, zorder=3)
        # plt.colorbar(scatter, ticks=np.unique(all_labels))
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=all_labels, cmap='tab10', edgecolors='black', linewidth=0.5, zorder=3)
        fig.colorbar(scatter, ticks=np.unique(all_labels))
        plt.title('t-SNE 3D visualization of latent space')
        plt.savefig(filename)
        # plt.show()


def visualize_pseudo_sample(all_feas, all_labels, pseudo_label_mask, epoch, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=5.0,
                learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(all_feas)

    filename = './ScatterPlots/pseudo_tsne_{}.png'.format(epoch)
    filename2 = './ScatterPlots/labels_tsne{}.png'.format(epoch)

    unique_classes = np.unique(all_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    if n_components == 2:
        plt.figure(figsize=(10, 7))
        plt.grid(True, linestyle='--', color="white", linewidth=1.0, zorder=0)
        plt.gca().set_facecolor("#EAEAEA")
        plt.xticks(np.linspace(min(tsne_results[:, 0]), max(tsne_results[:, 0]), num=5), [])
        plt.yticks(np.linspace(min(tsne_results[:, 1]), max(tsne_results[:, 1]), num=5), [])

        # Iterate over classes and plot them
        unique_classes = np.unique(all_labels)
        for class_idx in unique_classes:
            class_mask = all_labels == class_idx
            normal_samples = class_mask & ~pseudo_label_mask
            pseudo_samples = class_mask & pseudo_label_mask

            # Plot normal samples with class colors
            plt.scatter(tsne_results[normal_samples, 0], tsne_results[normal_samples, 1], color='darkgray', alpha=0.6, cmap='tab10', edgecolors='black', linewidth=0.5, zorder=3)

            # Plot pseudo-labeled samples in red
            plt.scatter(tsne_results[pseudo_samples, 0], tsne_results[pseudo_samples, 1], color=[class_color_map[class_idx]] * int(np.sum(pseudo_samples).item()),
                        edgecolors='black', linewidth=0.5, cmap='tab10', alpha=0.8, zorder=5)

        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        # plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        plt.figure(figsize=(10, 7))
        plt.grid(True, linestyle='--', color="white", linewidth=1.0, zorder=0)
        plt.gca().set_facecolor("#EAEAEA")
        plt.xticks(np.linspace(min(tsne_results[:, 0]), max(tsne_results[:, 0]), num=5), [])
        plt.yticks(np.linspace(min(tsne_results[:, 1]), max(tsne_results[:, 1]), num=5), [])

        # Iterate over classes and plot them
        unique_classes = np.unique(all_labels)
        for class_idx in unique_classes:
            class_mask = all_labels == class_idx
            normal_samples = class_mask & ~pseudo_label_mask
            pseudo_samples = class_mask & pseudo_label_mask

            plt.scatter(tsne_results[class_mask, 0], tsne_results[class_mask, 1], color=[class_color_map[class_idx]] * int(np.sum(class_mask).item()),
                        edgecolors='black', linewidth=0.5, cmap='tab10', alpha=0.8, zorder=5)

        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        # plt.legend()
        plt.tight_layout()
        plt.savefig(filename2)
        plt.close()


def visualize_latent_space_with_prototype(all_feas, all_labels, classifier_weights,
                                          epoch, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    filename = './ScatterPlots/tsne_{}.png'.format(epoch)

    all_feas_np = all_feas.cpu().numpy()  # (num_samples, num_features)
    all_labels_np = all_labels.cpu().numpy()  # (num_samples,)
    classifier_weights_np = classifier_weights.cpu().numpy()  # (num_classes, num_features)

    combined_data = np.vstack([all_feas_np, classifier_weights_np])

    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    tsne_results = tsne.fit_transform(combined_data)

    tsne_features = tsne_results[:len(all_feas_np), :]
    tsne_weights = tsne_results[len(all_feas_np):, :]

    if n_components == 2:
        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=all_labels_np, cmap='tab10', alpha=0.5,
                              label="Features")
        
        for i, (x, y) in enumerate(tsne_weights):
            plt.text(x, y, str(i), fontsize=12, fontweight='bold', color='red', ha='center', va='center')

        plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], c='red', marker='X', s=100, label="Classifier Weights")

        plt.colorbar(scatter, ticks=np.unique(all_labels_np))
        plt.title('t-SNE 2D visualization of Normalized Features and Classifier Weights')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig(filename)
        plt.show()


def visualize_prototype(prototype1, prototype2, n_components=2, perplexity=30, learning_rate=20, n_iter=1000):
    file_name = './visual/prototypes.png'
    prototype1_np = prototype1.numpy()
    prototype2_np = prototype2.numpy()
    label_np = np.arange(prototype1_np.shape[0])

    combined_data = np.vstack([prototype1_np, prototype2_np])

    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=5.0, learning_rate=learning_rate, n_iter=n_iter)
    tsne_results = tsne.fit_transform(combined_data)

    tsne_prototype1 = tsne_results[:len(prototype1_np), :]
    tsne_prototype2 = tsne_results[len(prototype1_np):, :]

    plt.figure(figsize=(10, 8))

    plt.scatter(tsne_prototype1[:, 0], tsne_prototype1[:, 1], c=label_np, marker='X', s=100, label="prototype")
    for i, (x, y) in enumerate(tsne_prototype1):
        plt.text(x, y, str(label_np[i]), fontsize=9, ha='right', color='blue')

    plt.scatter(tsne_prototype2[:, 0], tsne_prototype2[:, 1], c=label_np, marker='^', s=100, label="refined_prototype")
    for i, (x, y) in enumerate(tsne_prototype2):
        plt.text(x, y, str(label_np[i]), fontsize=9, ha='left', color='red')

    plt.legend()
    plt.savefig(file_name)


def visualize_latent_space_with_prototype_refined_prototype(all_feas, all_labels, classifier_weights, refined_classifier_weights,
                                          epoch, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    filename = './ScatterPlots/tsne_{}.png'.format(epoch)

    all_feas_np = all_feas.cpu().numpy()  # (num_samples, num_features)
    all_labels_np = all_labels.cpu().numpy()  # (num_samples,)
    classifier_weights_np = classifier_weights.cpu().numpy()  # (num_classes, num_features)
    refined_classifier_weights_np = refined_classifier_weights.cpu().numpy()  # (num_classes, num_features)

    print(f"classifier_weights_np.shape: {classifier_weights_np.shape}")
    print(f"refined_classifier_weights_np.shape: {refined_classifier_weights_np.shape}")

    all_feas_np = normalize(all_feas_np, axis=1)
    classifier_weights_np = normalize(classifier_weights_np, axis=1)
    refined_classifier_weights_np = normalize(refined_classifier_weights_np, axis=1)

    combined_data = np.vstack([all_feas_np, classifier_weights_np, refined_classifier_weights_np])

    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    tsne_results = tsne.fit_transform(combined_data)

    tsne_features = tsne_results[:len(all_feas_np), :]
    tsne_weights1 = tsne_results[len(all_feas_np):len(all_feas_np) + len(classifier_weights_np), :]
    tsne_weights2 = tsne_results[len(all_feas_np) + len(classifier_weights_np):, :]  # refined

    if n_components == 2:
        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=all_labels_np, cmap='tab10', alpha=0.5,
                              label="Features")

        # for i, (x, y) in enumerate(tsne_weights1):
        #     plt.text(x, y, str(i), fontsize=12, fontweight='bold', color='red', ha='center', va='center')

        plt.scatter(tsne_weights1[:, 0], tsne_weights1[:, 1], c='red', marker='X', s=100, label="Classifier Weights")
        plt.scatter(tsne_weights2[:, 0], tsne_weights2[:, 1], c='red', marker='^', s=100, label="Classifier Weights")

        plt.colorbar(scatter, ticks=np.unique(all_labels_np))
        plt.title('t-SNE 2D visualization of Normalized Features and Classifier Weights')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig(filename)


def visualize_sim(feature1, feature2, cycle, iter_num=0):
    filename = './ScatterPlots/cosine_cycle{}_iter{}.png'.format(cycle, iter_num)

    fea1_np = feature1.cpu().numpy()
    fea2_np = feature2.cpu().numpy()
    cos_sim = cosine_similarity(fea1_np, fea2_np)

    plt.figure(figsize=(8, 6))
    plt.imshow(cos_sim, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity between Features')
    plt.xlabel('Feature 2')
    plt.ylabel('Feature 1')
    plt.savefig(filename)


def visualize_tsne(feature1, feature2, cycle, iter_num=0, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    filename = './ScatterPlots/tsne_cycle{}_iter{}.png'.format(cycle, iter_num)

    fea1_np = feature1.cpu().numpy()
    fea2_np = feature2.cpu().numpy()

    normalized_fea1 = normalize(fea1_np, axis=1)
    normalized_fea2 = normalize(fea2_np, axis=1)

    # combined_fea = np.vstack([fea1_np, fea2_np])
    combined_fea = np.vstack([normalized_fea1, normalized_fea2])

    # combined_fea = normalize(combined_fea, axis=1)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    tsne_results = tsne.fit_transform(combined_fea)

    tsne_fea1 = tsne_results[:len(fea1_np), :]
    tsne_fea2 = tsne_results[len(fea1_np):, :]

    if n_components == 2:
        plt.figure(figsize=(10, 8))

        plt.scatter(tsne_fea1[:, 0], tsne_fea1[:, 1], c='red', marker='X', s=100, label="Classifier Weights")
        plt.scatter(tsne_fea2[:, 0], tsne_fea2[:, 1], c='blue', marker='^', s=100, label="Classifier Weights")

        plt.title('t-SNE 2D visualization of Normalized Features and Classifier Weights')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig(filename)

    return None


def visualize_boundary_sample(all_feas, all_preds, all_preds_refined, all_labels, classifier_weights):
    all_feas_np = all_feas.numpy()
    all_labels_np = all_labels.numpy()
    all_preds_np = torch.argmax(all_preds, axis=1).numpy()
    all_preds_refined_np = torch.argmax(all_preds_refined, axis=1).numpy()
    classifier_weights_np = classifier_weights.cpu().numpy()

    all_feas_np = normalize(all_feas_np, axis=1)
    classifier_weights_np = normalize(classifier_weights_np, axis=1)

    incorrect_a_correct_b = np.where((all_preds_np != all_labels_np) & (all_preds_refined_np != all_labels_np))[0]

    tsne = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack([all_feas_np, classifier_weights_np])

    embedding_2d = tsne.fit_transform(combined_data)

    tsne_features = embedding_2d[:len(all_feas_np), :]
    tsne_weights = embedding_2d[len(all_feas_np):]

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c='lightgray', label='All Samples', alpha=0.6, s=50)

    plt.scatter(tsne_features[incorrect_a_correct_b, 0], tsne_features[incorrect_a_correct_b, 1],
                color='red', label='Incorrect Samples', edgecolor='black', s=50)
    plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], c='blue', marker='X', label='Classifier Weights', s=50)
    plt.legend()
    plt.title('boundary sample')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('./visual/boundary_sample_prototype.png')


def save_gif():
    frames = []
    imgs = sorted(os.listdir("./ScatterPlots"))

    for im in imgs:
        new_frame = Image.open("./ScatterPlots/" + im)
        frames.append(new_frame)

    frames[0].save("latentspace.gif", format="GIF",
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)


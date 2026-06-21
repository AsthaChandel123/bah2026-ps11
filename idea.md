PROBLEM STATEMENT 11

Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data
Description
Satellite remote sensing systems acquire Earth observation data using different sensors such as optical, multispectral, hyperspectral, Synthetic Aperture Radar (SAR), and elevation-based sensors. Each sensor captures different physical characteristics of the Earth surface. Optical and multispectral images provide rich visual and spectral information, while SAR images provide structural information and can operate under cloud cover and night-time conditions.

With the rapid growth of satellite image archives, content-based retrieval of relevant images has become an important requirement. Traditional metadata-based search may not be sufficient when similar land-cover or land-use regions are observed under different seasons, acquisition conditions, or sensor modalities. Therefore, this problem focuses on developing a multi-modal satellite image retrieval system for both same-modal and cross-modal retrieval.

Objectives
To develop an efficient satellite image retrieval framework that retrieves semantically similar remote sensing images from the same modality as well as from different modalities. The system should learn a common representation space where images with similar land-cover, land-use, or scene-level content are close to each other, irrespective of sensor modality.

Same-modal retrieval, such as optical-to-optical, SAR-to-SAR, and multispectral-to-multispectral retrieval.
Cross-modal retrieval, such as optical-to-SAR, SAR-to-optical, optical-to-multispectral, and multispectral-to-optical retrieval.
Ranking of the top-5 and top-10 most relevant images for each query image.
Efficient retrieval with low average retrieval time per query.
Expected Outcomes
A retrieval system that accepts a query satellite image from one modality and returns a ranked list of the most relevant images from a gallery database. The gallery may contain images from the same modality or different modalities. The final system should report top-5 and top-10 retrieval results along with retrieval time required per query.

Dataset Required
Multi-sensor remote sensing image data consisting of two or more aligned or semantically associated modalities. The dataset may include optical RGB images, multispectral images, SAR images, and optional land-cover or land-use labels. Each sample may correspond to the same or nearby geographic location observed using different sensors. Labels or metadata may be used for evaluating semantic relevance among retrieved images.

Suggested Tools/Technologies
Both classical machine learning and deep learning based techniques can be considered. Participants may use CNNs, Vision Transformers, Siamese or triplet networks, metric learning, contrastive learning, self-supervised learning, multi-modal representation learning, foundation model based feature extraction, or efficient vector search libraries such as FAISS. Use of pre trained model or foundation models are also allowed.

Expected Solution / Steps to be followed to achieve the objectives
Probable steps for data preparation and retrieval pipeline

Prepare paired or semantically associated multi-sensor satellite image data for training and evaluation.
Apply preprocessing, normalization, resizing, and modality-specific handling for optical, multispectral, and SAR images.
Define query and gallery sets for same-modal and cross-modal retrieval evaluation.
Probable steps for data preparation and retrieval pipeline

Select or design a feature extraction backbone suitable for remote sensing images.
Learn modality-specific or shared embedding using supervised, self-supervised, contrastive, or metric learning approaches.
Align features from different modalities into a common embedding space.
Generate compact image descriptors for all query and gallery images.
Probable steps for retrieval and ranking

Compute similarity between query and gallery descriptors using cosine similarity, Euclidean distance, or other suitable measures.
Rank gallery images according to similarity score.
Return top-5 and top-10 retrieved images for each query.
Measure average retrieval time per query, including feature matching or similarity search time.
Evaluation Criteria
The evaluation will be carried out for both same-modal retrieval and cross-modal retrieval. For each query image, the submitted system should produce top-5 and top-10 retrieved images from the gallery. The retrieved images will be compared with the relevant ground-truth set based on semantic class, geographic correspondence, or predefined relevance labels.

F1-score@5 for same-modal retrieval.
F1-score@10 for same-modal retrieval.
F1-score@5 for cross-modal retrieval.
F1-score@10 for cross-modal retrieval.
Average retrieval time per query.
The final ranking may consider both retrieval accuracy and computational efficiency. Higher F1-scores and lower retrieval time will be preferred. Cross-modal retrieval may be given additional importance because it is more challenging than same-modal retrieval.

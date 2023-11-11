# Literature
The study "Increasing crop rotational diversity can enhance cereal yields" by Monique E. Smith et al. explores the impact of crop rotational diversity (CRD) on cereal yields, considering factors like external nitrogen fertilization levels and functional crop groups​​.

Key findings include:

Impact of CRD on Yields: Using data from 32 long-term experiments across Europe and North America, the study reveals that increasing CRD, both in terms of species diversity and functional richness, enhances grain yields of small grain cereals and maize. This yield benefit grows over time, although winter-sown small grain cereals exhibit a yield decline at the highest species diversity levels. Diversification is particularly beneficial for cereals with low external nitrogen input, such as maize, reducing reliance on nitrogen fertilizers and decreasing greenhouse gas emissions and nitrogen pollution​​.

Diverse Rotations and Reduced Inputs: Diverse crop rotations require less fertilization and crop protection inputs, easing agricultural pressure on climate, soil, and biodiversity. The study indicates that CRD results in higher yields when nitrogen input is low, particularly when legumes are included. However, the extent to which diverse rotations can maintain yields and offset reduced fertilizer inputs over time for different grain crops remains unclear​​.

Crop Reactions to Functional Groups: The inclusion of certain functional groups, such as annual broadleaf and legume crops, benefits the production of winter small grain cereals. For maize, including annual legume and ley crops is beneficial, but not annual broadleaf crops. These findings suggest that ecological complementarity mechanisms play a significant role in increasing yields​​.

CRD and Crop Protection: Increasing CRD likely leads to an underestimation of its positive effects on crop protection, as chemical weed control needs often reduce with higher CRD. The diversity in root depth, architecture, and resource needs among crops leads to complementary nutrient and water uptake, enhancing soil organic carbon content and improving soil structure, nutrient stocks, water retention, and yields​​.

CRD and External Nitrogen Input: Maize yields increase more significantly with CRD under low rather than high external nitrogen conditions. The study suggests that the biophysical source of production benefits varies among crops, depending on both crop diversity and functional group richness at different nitrogen input levels​​.

General Trends and Variability: The study acknowledges large variations in growing conditions and management across experiments, indicating a general trend of high grain yield benefits from CRD. It suggests that CRD effects should be evaluated case-by-case, and individual farmers must assess its applicability​​.

Implications and Future Directions: The study highlights the need for a more diversified agricultural approach to support crop yields while reducing environmental and societal costs. This approach could also reduce food system vulnerability to climate change and stressful weather​​.

Overall, the study presents a comprehensive analysis of how increasing crop rotational diversity can enhance cereal yields, emphasizing the importance of incorporating diverse crop species and functional groups into agricultural practices for sustainable and efficient crop production.

The paper "Crop Rotation Modeling for Deep Learning-Based Parcel Classification from Satellite Time Series" by Quinton and Landrieu focuses on the crucial yet often overlooked role of annual crop rotations in agricultural optimization and crop type mapping. The main purpose of this paper is to model the inter- and intra-annual agricultural dynamics of yearly parcel classification using a deep learning approach, leveraging the increasing quantity of annotated satellite data. This approach led to an improvement of over 6.3% in mean Intersection over Union (mIoU) over the state-of-the-art crop classification and a reduction of over 21% in error rate. Additionally, the paper introduces the first large-scale multi-year agricultural dataset with over 300,000 annotated parcels​​.

The authors employed several innovative methods for their research and analysis:

Dataset: The study used a dataset based on parcels within the 31TFM Sentinel-2 tile, covering an area of 110 × 110 km² in the South East of France. The dataset included over 54,000 ha of corn and 30,000 ha of wheat, with meadows being the most common crop type. The data spanned three years (2018-2020) and involved Sentinel-2 level 2A images. Stable parcels with minor contour changes over the years were selected for the study​​.

Pixel-Set and Temporal Attention Encoders: The Pixel Set Encoder (PSE) and Temporal Attention Encoder (TAE) were used. The PSE is a spatio-spectral encoder that generates descriptors of the spectral distribution of observations within a parcel, and the TAE is a temporal sequence encoder based on language processing literature, adapted for processing Satellite Image Time Series (SITS)​​.

Multi-Year Modeling: The study introduced a modification to the PSE+LTAE network to model crop rotation. This involved augmenting the spatio-temporal descriptors by concatenating the sum of one-hot-encoded labels for the previous two years, allowing the model to simultaneously learn inter-annual crop rotations and intra-annual evolution of parcels’ spectral statistics​​.

Training Protocol: A mixed-year training protocol was used, training a single model with parcels from all available years. This approach aimed to create richer and more resilient descriptors by exposing the model to data from different years, each with varying meteorological conditions​​.

Cross-Validation: The data was divided into 5 folds for cross-validation, with each fold being trained on 3 folds and using the last fold for calibration and model selection. This approach ensured that parcels did not appear in multiple folds for different years, avoiding data contamination and self-correlation​​.

The key findings and statistics in the paper are:

Performance Improvement: The mixed-year model (Mmixed) significantly outperformed the specialized models, demonstrating better generalizability and precision. This model obtained an mIoU of 84.7% and an overall accuracy of 98.1% on the training set​​​​.

Influence of Crop Rotation Modeling: The proposed model (Mdec) achieved higher performance than the single-year model (Msingle) with a 6.3% mIoU gap. This improvement was attributed to the model's ability to consider both the current year’s observations and the influence of past cultivated crops​​.

Overall Impact: Training a deep learning model from multi-year observations improved its ability to generalize and resulted in better precision across different crop types, especially for those with strong temporal structures​​.

In summary, this paper made significant strides in improving automated crop type mapping by incorporating multi-year data and crop rotation dynamics into a deep learning framework, demonstrating notable improvements in classification accuracy and precision.

The main purpose of the paper "Crop mapping from image time series: Deep learning with multi-scale label hierarchies" by Turkoglu et al. is to map agricultural crops by classifying satellite image time series using a deep learning network architecture that exploits a hierarchical tree structure of crop type labels. This approach significantly improves the mapping of rare crop types by allowing the model to predict three labels at different levels of granularity for each pixel, enhancing classification performance at the fine-grained level​​​​.

The methods used in the research include:

Hierarchical Deep Learning Network Architecture: The architecture combines convolutional and recurrent neural networks to exploit a tree-structured label hierarchy, encode image data, and represent time series. The network predicts successively finer label resolutions in the hierarchical tree, and a label refinement module refines these predictions using a Convolutional Neural Network (CNN) that exploits correlations between labels across the hierarchy​​.

ZueriCrop Dataset: The research introduces the ZueriCrop dataset, which is based on farm census data from the Swiss Federal Office for Agriculture. It includes 116,000 field instances from the Swiss Cantons of Zurich and Thurgau, covering 48 different classes with a highly imbalanced class distribution. The dataset contains multi-spectral Sentinel-2 Level-2A bottom-of-atmosphere reflectance images collected over a 50 km × 48 km area​​.

Evaluation Methodology: The study employs 5-fold leave-one-out cross-validation, dividing the dataset into five geographically disjoint strips and using four as the training set and one as the test set. Classification performance is evaluated using metrics like overall accuracy, average per-class precision, recall, and F1-score​​.

Key data and statistics mentioned in the paper include:

Dataset Overview: ZueriCrop spans 48 crop classes and 28,000 multi-temporal image patches from Sentinel-2. The dataset's area covers 50 km × 48 km in the Swiss cantons of Zurich and Thurgau with 116,000 individual fields​​.

Performance Metrics: The hierarchical ms-convSTAR significantly outperforms other models, increasing mean class precision by more than 11 percentage points and mean class recall by more than 10 percentage points. The F1-score increases by more than 11 percentage points compared to the baseline. This improvement is especially notable in the classification of less frequent classes​​.

Comparison with Baselines: The hierarchical ms-convSTAR outperforms all baseline models on all performance metrics. Data augmentation, a baseline method, improves the F1-score by 2.9 percentage points compared to the baseline convSTAR but decreases overall accuracy significantly by 2.3 percentage points​​.

In summary, this paper presents a novel approach to crop mapping from satellite image time series, leveraging hierarchical label structures and deep learning techniques to improve classification accuracy, especially for rare crop types. The use of the ZueriCrop dataset provides a realistic scenario for testing and demonstrates the effectiveness of the proposed method over traditional approaches.

The study "Pixel-based yield mapping and prediction from Sentinel-2 using spectral indices and neural networks" by Perich et al. aims to map and predict crop yield on a large scale, particularly in smaller scaled agricultural settings like Switzerland. The study focuses on using high-resolution Sentinel-2 (S2) imagery for within-field crop yield modeling at the pixel level, employing a mix of spectral indices, raw satellite reflectance data, and a recurrent neural network (RNN) approach​​.

Key components of the study include:

Yield Data: The study used yield data from a large farm in western Switzerland, focusing on cereal crops like winter wheat, winter barley, and triticale from 2017 to 2021. The data was pre-processed and rasterised to a 10m raster for modeling, resulting in 54,098 pixels with yield information for the cereal crops (CR) dataset and 20,170 pixels for the winter wheat (WW) subset​​.

Sentinel-2 Data: All available S2 scenes from January 2017 to December 2021 were downloaded and pre-processed. The analysis excluded bands affected by atmospheric disturbances and used the Scene Classification Layer (SCL) to filter out clouds, cloud shadows, dark areas, and defective pixels. This resulted in a comprehensive dataset for modeling, with data from 134 S2 scenes retained for analysis​​.

Methods: Four different models were compared for their applicability in crop yield modeling and prediction: two based on spectral indices (partial integral at peak GCVI and smoothed NDVI), one using all available spectral bands of S2 (four S2 scenes method), and an RNN model. The study aimed to assess the robustness, accuracy, and data processing needs of these methods​​.

Results: The models performed best when using data from all years, with R2 values up to 0.88 and relative RMSE up to 10.49%. However, performance was poor when predicting on unseen data years, particularly for years with unknown weather patterns. The RNN showed similar performance across different input S2 time series, with the cloudy time series exhibiting the best performance in some cases. The four S2 scenes method and the RNN had the lowest RMSE values, indicating their effectiveness​​​​​​.

Cross-Year Performance: The models struggled to predict crop yield accurately for unseen data years. Negative R2 values were recorded for the WW subset and most of the CR dataset, indicating that predictions were far from the mean of the test data. Only the four S2 scenes and RNN methods showed positive but low R2 values for the holdout years 2017–2019​​.

Discussion: The study demonstrates that precise yield modeling and prediction are feasible using high-resolution S2 imagery in smaller scaled agricultural settings. It highlights the potential for such satellite-based approaches to assist farmers in making informed management decisions and for policymakers to assess yield gaps and food security. The study also discusses the need for calibration of yield data and suggests that validation studies in different regions are necessary for broader application​​.

In summary, the paper presents a comprehensive approach to crop yield modeling using Sentinel-2 data, demonstrating the potential of high-resolution satellite imagery in precision agriculture, especially in smaller agricultural settings. The combination of different modeling methods, including RNNs and spectral indices, provides insights into the possibilities and limitations of current satellite-based yield prediction technologies.

The study "Mapping Crop Rotation by Using Deeply Synergistic Optical and SAR Time Series" by Liu et al. aimed to improve the mapping of crop rotation patterns, a crucial aspect of agricultural management impacting food security and agro-ecosystem sustainability. The main focus was on developing a hybrid deep learning architecture, the Crop Rotation Mapping (CRM) model, which synergizes Synthetic Aperture Radar (SAR) and optical time series data for mapping crop rotations. This approach showed significant improvement over traditional methods, particularly for complex rotation types like fallow-single rice and crayfish-single rice, with an accuracy greater than 0.85 and a gain of four points in overall accuracy compared to ablation models​​​​.

Key aspects of the study include:

Ground-Truth Data Collection: Field surveys were conducted to gather ground truth data on crop rotation for three years (2018–2020). A total of 689 ground samples were collected, with additional non-cropland samples added through visual interpretation on Google Earth. This data helped classify six main rotation types and four non-rotation classes for model training​​.

CRM Architecture: The CRM model consists of two streams, each designed for SAR and optical time series data. These streams use a one-dimensional CNN (Conv1D) and an LSTM with an attention mechanism (AttLSTM). The CNN extracts hierarchical rotation features, while the LSTM depicts temporal crop growing patterns. The concatenated feature vectors from both streams are then processed through fully connected layers to produce the final rotation decision​​.

Experiment Design and Comparative Analysis: The study compared the performance of the CRM model against four competing models (RF(S2), LSTM(S2), CRM(S2), and CRM(NoAtt)) using different vegetation indices (NDVI, EVI, kNDVI) as inputs. This comparison aimed to evaluate the CRM's effectiveness against classical and advanced models and to analyze the sensitivity of various vegetation indices as model inputs​​.

Model Evaluation: The model's performance was evaluated using accuracy indices and a confusion matrix, including global and per-class accuracy, kappa coefficient, and F1 score. These metrics helped determine the model's ability to accurately predict crop rotation types​​.

Results and Sensitivity Analysis: The CRM model, using kNDVI as the vegetation index, achieved the highest accuracy (0.876) among all tested models and vegetation indices. CRM outperformed the mono-source competing methods (RF(S2), LSTM(S2), and CRM(S2)) with an average accuracy of 0.87. The CRM also showed superior capabilities when combining multiple satellite sources and utilizing the attention mechanism, which assigned greater weights to more important timestamps​​.

In conclusion, the study demonstrated the effectiveness of the CRM model in mapping crop rotations using a combination of SAR and optical time series data. The model's ability to leverage complementary signals from different sensors and the use of deep learning frameworks enabled it to capture complex intra-annual rotation information suitable for diverse crop types and flexible farming practices. The research highlights the potential of advanced deep learning techniques in enhancing the precision and utility of agricultural mapping systems, especially in the context of dynamic crop rotation mapping​​.

The paper "Crop switching can enhance environmental sustainability and farmer incomes in China" by Xie et al. explores the impact of crop switching on various sustainability dimensions in China. The main purpose of the study is to evaluate if strategic crop switching can lead to more sustainable cropping systems and to understand the necessity of coordinated actions to avoid trade-offs between different sustainability objectives.

Key findings and insights of the study include:

Multi-Dimensional Sustainability Challenge: The study acknowledges that food-system sustainability is a complex, multi-dimensional challenge. Despite China's efforts to enhance production efficiency and reduce environmental impacts, there is limited understanding of the potential of crop switching in achieving sustainable cropping systems and the need for coordinated action to avoid trade-offs​​.

Potential of Crop Switching: The research shows considerable potential for crop switching to enhance sustainable development. The study finds that prioritizing a single sustainability objective could significantly reduce environmental impacts (e.g., water use, GHG emissions, fertilizer and pesticide usage) without expanding cropland or reducing crop production or farmer incomes. However, this could lead to trade-offs between different sustainability dimensions and regions​​.

Central Coordination and Co-Benefits: In scenarios where China's central government leads coordination (G3 simulation), the study finds that crop switching can simultaneously benefit multiple sustainability dimensions. This approach includes reductions in blue and green water use, GHG emissions, fertilizer and pesticide use, and increases in farmer incomes. The improvements range from 4.5% to 18.5% across various dimensions, indicating substantial co-benefits from centrally coordinated crop switching​​.

Spatially Detailed Solutions: The study emphasizes the importance of spatially detailed solutions tailored to local conditions and sustainability priorities. It points out specific regions where crop switching can lead to significant sustainability benefits. For example, shifts from maize to soybean, sugar beet, and rice in the Northeast Plain could reduce fertilizer and pesticide overuse and improve farmer incomes. Similarly, changes in crop patterns in other regions like the Yangtze River Plain and the North China Plain can contribute to more sustainable cropping patterns and alleviate regional issues like water scarcity and excessive fertilizer use​​.

The paper concludes that crop switching is a viable strategy to achieve sustainable development targets in China while improving farmer incomes and maintaining national production. It underscores the need for large-scale coordination and inter-ministry cooperation to realize these benefits, especially in a country like China with a centralized planning government. The study provides detailed, actionable evidence for policy-making and the implementation of sustainable agricultural interventions.

The paper "Enhanced agricultural sustainability through within-species diversification" by Li-Na Yang and colleagues explores the impact of within-species diversification in agriculture, focusing specifically on potato cultivation. The main purpose of the study is to investigate how diversification within a single species, like different potato varieties, can enhance agricultural sustainability. The study examines the effects on yield, disease resistance, soil health, and pathogen evolution.

Key methods used in the study include:

Field Experiments: Conducted in Yunnan province over two years, the experiments involved growing six potato cultivars, varying in agronomic traits and resistance to P. infestans (the cause of late blight), either alone or in mixtures involving 2-6 cultivars. The study used a randomized complete block design with three replicates​​.

Soil Analysis: In the Yunnan trials, soil nitrogen (N), phosphorus (P), potassium (K), and carbon contents were measured in plots planted with monocultures and different mixtures of potato varieties. This was to examine the relationship between the number of varieties per plot and soil nutrient levels​​.

Molecular and Biological Approaches: To understand the impact of host diversity on the evolution of P. infestans, 2049 single-spore isolates were genotyped using simple sequence repeat (SSR) markers. Additional tests were conducted on a subset of these isolates for aggressiveness and gene sequencing​​.

Significant findings from the study include:

Yield and Disease Resilience: There was a positive correlation between average yield and potato diversity. Increased diversity in potato varieties led to higher yields and more resilient crops against diseases like late blight​​.

Soil Health Improvements: Higher diversity in potato varieties correlated with increased total and alkaline hydrolysable soil nitrogen and organic soil carbon. This suggests that increased host diversity leads to higher soil fertility with less fluctuation in nutrient levels​​.

Pathogen Evolution: The study found that higher diversity in potato populations led to a more stable pathogen population structure, weaker host selection in host-pathogen interactions, and reduced aggressiveness in pathogen isolates. This indicates that increasing host diversity can impose disruptive selection on pathogen populations, potentially slowing down the evolution of more aggressive pathogen strains​​.

In summary, the study demonstrates that within-species diversification in agriculture, such as growing multiple varieties of potatoes, can enhance yield, improve soil health, and reduce the impact and evolution of diseases like late blight. These findings support the potential of within-species diversification as a sustainable agricultural practice.

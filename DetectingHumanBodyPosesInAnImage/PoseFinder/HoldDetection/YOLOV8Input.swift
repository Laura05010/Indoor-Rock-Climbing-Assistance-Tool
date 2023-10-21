//
//  YOLOV8Input.swift
//  PoseFinder
//
//  Created by Laura M Madrid on 2023-10-20.
//  Copyright © 2023 Apple. All rights reserved.
//

import CoreML
import Vision

/// - Tag: PoseNetInput
class YOLOV8Input: MLFeatureProvider {
    /// The name of the PoseNet model's input feature.
    ///
    /// You can see all the model's inputs and outputs, their names, and other information by selecting
    /// `PoseNetMobileNet075S16FP16.mlmodel` in the Project Navigator.
    private static let imageFeatureName = "image"

    /// The original image this feature provider uses as input to its consumers.
    var imageFeature: CGImage

    /// The size of the images this feature provider produces to its consumers.
    ///
    /// The feature provider resizes the `imageFeature` property by
    /// cropping and resizing it to this size in the `featureValue(for:)` method.
    let imageFeatureSize: CGSize

    var featureNames: Set<String> {
        return [YOLOV8Input.imageFeatureName]
    }

    init(image: CGImage, size: CGSize) {
        imageFeature = image
        imageFeatureSize = size
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == YOLOV8Input.imageFeatureName else {
            return nil
        }

        let options: [MLFeatureValue.ImageOption: Any] = [
            .cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
        ]

        return try? MLFeatureValue(cgImage: imageFeature,
                                   pixelsWide: Int(imageFeatureSize.width),
                                   pixelsHigh: Int(imageFeatureSize.height),
                                   pixelFormatType: imageFeature.pixelFormatInfo.rawValue,
                                   options: options)
    }
}


//
//  YOLOV8.swift
//  PoseFinder
//
//  Created by Laura M Madrid on 2023-10-20.
//  Copyright © 2023 Apple. All rights reserved.
//

import CoreML
import Vision

protocol YOLOV8Delegate: AnyObject {
    func yoloV8(_ yolov8: YOLOV8, didPredict predictions: YOLOV8Output)
}

class YOLOV8 {
    /// The delegate to receive the PoseNet model's outputs.
    weak var delegate: YOLOV8Delegate?

    /// The PoseNet model's input size.
    ///
    /// All PoseNet models available from the Model Gallery support the input sizes 257x257, 353x353, and 513x513.
    /// Larger images typically offer higher accuracy but are more computationally expensive. The ideal size depends
    /// on the context of use and target devices, typically discovered through trial and error.
    let modelInputSize = CGSize(width: 640, height:  640)

    /// The PoseNet model's output stride.
    ///
    /// Valid strides are 16 and 8 and define the resolution of the grid output by the model. Smaller strides
    /// result in higher-resolution grids with an expected increase in accuracy but require more computation. Larger
    /// strides provide a more coarse grid and typically less accurate but are computationally cheaper in comparison.
    ///
    /// - Note: The output stride is dependent on the chosen model and specified in the metadata. Other variants of the
    /// PoseNet models are available from the Model Gallery.
    let outputStride = 16

    /// The Core ML model that the PoseNet model uses to generate estimates for the poses.
    ///
    /// - Note: Other variants of the PoseNet model are available from the Model Gallery.
    private let YOLOV8MLModel: MLModel

    init() throws {
        YOLOV8MLModel = try hold_detection(configuration: .init()).model
    }

    /// Calls the `prediction` method of the PoseNet model and returns the outputs to the assigned
    /// `delegate`.
    ///
    /// - parameters:
    ///     - image: Image passed by the PoseNet model.
    func predict(_ image: CGImage) {
        DispatchQueue.global(qos: .userInitiated).async {
            // Wrap the image in an instance of PoseNetInput to have it resized
            // before being passed to the PoseNet model.
            let input = YOLOV8Input(image: image, size: self.modelInputSize)

            guard let prediction = try? self.YOLOV8MLModel.prediction(from: input) else {
                return
            }

            let yolov8Output = YOLOV8Output(prediction: prediction,
                                              modelInputSize: self.modelInputSize,
                                              modelOutputStride: self.outputStride)

            DispatchQueue.main.async {
            
                self.delegate?.yoloV8(self, didPredict: yolov8Output)
            }
        }
    }
}

//
//  ARViewContainer.swift
//  BodyPoseTracker
//
//  Created by Laura M Madrid on 2023-09-22.
//

import Foundation
import SwiftUI
import ARKit
import RealityKit

struct ARViewContainer: UIViewRepresentable {
    typealias UIViewType = ARView
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame:.zero, cameraMode: .ar, automaticallyConfigureSession: true)
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        return
    }
    
    
}

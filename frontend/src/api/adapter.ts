// src/api/adapter.ts
// Utility functions to map backend responses to frontend models without assuming exact schema.

/**
 * Convert backend job status string to a numeric pipeline step index.
 * Steps (example):
 * 0 - Not started
 * 1 - Uploaded (queued)
 * 2 - Processing
 * 3 - Generating report
 * 4 - Completed
 */
export const mapJobStatusToStep = (status: string): number => {
  switch (status) {
    case 'queued':
      return 1;
    case 'processing':
    case 'running':
      return 2;
    case 'generating_report':
      return 3;
    case 'completed':
      return 4;
    case 'failed':
      return -1;
    default:
      return 0;
  }
};

/**
 * Normalise prediction payload – fields may be missing.
 */
export interface PredictionResult {
  diagnosis?: string;
  confidence?: number;
  model?: string;
  processingTime?: string;
}

export const mapPredictionPayload = (payload: any): PredictionResult => {
  return {
    diagnosis: payload.prediction_label ?? payload.label,
    confidence: payload.confidence_score ?? payload.confidence,
    model: payload.model_version ?? payload.model,
    processingTime: payload.processing_time ?? payload.time,
  };
};

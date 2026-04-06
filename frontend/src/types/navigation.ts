import type { PredictResponse } from "../api";

export type ResultsLocationState = {
  result: PredictResponse;
  fileName: string;
  /** In-memory only; Results page creates / revokes object URL. */
  file: File;
};

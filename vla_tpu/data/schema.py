from dataclasses import dataclass


@dataclass
class ObservationBatch:
    images: object
    state: object
    instruction_tokens: object
    actions: object

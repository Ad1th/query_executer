from dataclasses import dataclass, asdict
from typing import List

@dataclass
class QueryParameter:
    name: str
    data_type: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return asdict(self)


@dataclass
class ReadyQuery:
    query: str
    variables: list[dict]


class BenchmarkQuery:
    def __init__(self, query: str, parameters: List[QueryParameter], database: str, benchmark: str, name: str):
        self.query = query
        self.parameters = parameters
        self.database = database
        self.name = name
        self.benchmark = benchmark
        self.name = name

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            query=data['query'],
            parameters=[QueryParameter.from_dict(p) for p in data['parameters']],
            database=data['database'],
            benchmark=data['benchmark'],
            name=data['name']
        )

    def to_dict(self):
        return {
            'query': self.query,
            'parameters': [p.to_dict() for p in self.parameters],
            'database': self.database,
            'benchmark': self.benchmark,
            'name': self.name,
        }


@dataclass
class QueueState:
    queries_in_queue: int
    total_executed_batch: int


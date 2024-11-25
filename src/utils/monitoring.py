import boto3
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MetricData:
    timestamp: float
    value: float
    dimensions: Dict[str, str]

class CloudWatchMonitor:
    def __init__(self, namespace: str = 'MLTraining'):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = namespace
        
    def log_metric(self, metric_name: str, value: float, dimensions: Dict[str, str] = None):
        """Log a metric to CloudWatch"""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': 'None',
                'Timestamp': time.time()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
                
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
            
        except Exception as e:
            print(f"Error logging metric {metric_name}: {e}")
            
    def create_dashboard(self, dashboard_name: str, metrics: list):
        """Create a CloudWatch dashboard for monitoring"""
        try:
            widgets = []
            for metric in metrics:
                widget = {
                    'type': 'metric',
                    'properties': {
                        'metrics': [[self.namespace, metric]],
                        'period': 300,
                        'stat': 'Average',
                        'title': f'{metric} Over Time'
                    }
                }
                widgets.append(widget)
                
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=str({'widgets': widgets})
            )
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
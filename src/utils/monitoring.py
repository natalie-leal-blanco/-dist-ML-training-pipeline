import boto3
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json

@dataclass
class MetricData:
    timestamp: float
    value: float
    dimensions: Dict[str, str]

class CloudWatchMonitor:
    def __init__(self, namespace: str = 'MLTraining'):
        """Initialize CloudWatch monitoring."""
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = namespace
        self._dashboard_metrics = []
    
    def log_metric(self, metric_name: str, value: float, dimensions: Optional[Dict[str, str]] = None):
        """Log a metric to CloudWatch."""
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
            print(f"Logged metric: {metric_name} = {value}")
            
        except Exception as e:
            print(f"Error logging metric {metric_name}: {e}")
    
    def create_dashboard(self, dashboard_name: str, metrics: List[Dict[str, Any]]):
        """Create a CloudWatch dashboard for monitoring."""
        try:
            widgets = []
            for idx, metric in enumerate(metrics):
                widget = {
                    'type': 'metric',
                    'x': (idx % 2) * 12,
                    'y': (idx // 2) * 6,
                    'width': 12,
                    'height': 6,
                    'properties': {
                        'metrics': [[self.namespace, metric['name']]],
                        'period': metric.get('frequency', 300),
                        'stat': 'Average',
                        'title': f"{metric['name'].replace('_', ' ').title()}"
                    }
                }
                widgets.append(widget)
            
            dashboard_body = json.dumps({'widgets': widgets})
            
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=dashboard_body
            )
            print(f"Created dashboard: {dashboard_name}")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            raise e
    
    def create_alarms(self, alarms: List[Dict[str, Any]]):
        """Create CloudWatch alarms for metrics."""
        try:
            for alarm in alarms:
                self.cloudwatch.put_metric_alarm(
                    AlarmName=f"MLTraining_{alarm['metric']}",
                    MetricName=alarm['metric'],
                    Namespace=self.namespace,
                    Period=alarm.get('window', 300),
                    EvaluationPeriods=2,
                    Threshold=float(alarm['condition'].split()[1]),
                    ComparisonOperator='GreaterThanThreshold' 
                        if '>' in alarm['condition'] else 'LessThanThreshold',
                    Statistic='Average',
                    ActionsEnabled=True
                )
                print(f"Created alarm for: {alarm['metric']}")
                
        except Exception as e:
            print(f"Error creating alarms: {e}")
            raise e

    def list_metrics(self) -> List[str]:
        """List all available metrics in the namespace."""
        try:
            response = self.cloudwatch.list_metrics(
                Namespace=self.namespace
            )
            return list(set(metric['MetricName'] for metric in response['Metrics']))
        except Exception as e:
            print(f"Error listing metrics: {e}")
            return []
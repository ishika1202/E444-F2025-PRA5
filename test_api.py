import requests
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Your API endpoint
API_URL = "http://pra5ece444-env.eba-nwnz2nih.ca-central-1.elasticbeanstalk.com/predict"

# Test cases: 2 fake news and 2 real news
test_cases = {
    "fake_news_1": "Scientists confirm the earth is flat and NASA has been lying to us for decades",
    "fake_news_2": "Miracle cure discovered: drinking bleach cures all diseases including cancer",
    "real_news_1": "The stock market experienced volatility today as investors reacted to new economic data",
    "real_news_2": "Researchers publish new findings on climate change in peer-reviewed journal"
}

def make_request(message):
    """Make a single API request and return the latency in milliseconds"""
    start_time = time.time()
    try:
        response = requests.post(
            API_URL,
            json={"message": message},
            timeout=10
        )
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "latency": latency,
                "prediction": result.get("label", "N/A"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "latency": latency,
                "error": f"Status {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "success": False,
            "latency": latency,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def test_endpoint(test_name, message, num_requests=100):
    """Test an endpoint with multiple requests and save results to CSV"""
    print(f"\nTesting: {test_name}")
    print(f"Message: {message[:50]}...")
    print(f"Making {num_requests} requests...")
    
    results = []
    latencies = []
    
    for i in range(num_requests):
        result = make_request(message)
        results.append(result)
        latencies.append(result["latency"])
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests")
    
    # Save to CSV
    csv_filename = f"{test_name}_results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['request_number', 'timestamp', 'latency_ms', 'success', 'prediction', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, result in enumerate(results, 1):
            writer.writerow({
                'request_number': i,
                'timestamp': result['timestamp'],
                'latency_ms': f"{result['latency']:.2f}",
                'success': result['success'],
                'prediction': result.get('prediction', ''),
                'error': result.get('error', '')
            })
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    median_latency = np.median(latencies)
    
    print(f"  Results saved to {csv_filename}")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Median latency: {median_latency:.2f} ms")
    
    return {
        'name': test_name,
        'latencies': latencies,
        'avg': avg_latency,
        'min': min_latency,
        'max': max_latency,
        'median': median_latency
    }

def create_boxplots(all_results):
    """Create boxplots for all test cases"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for boxplot
    data = [result['latencies'] for result in all_results]
    labels = [result['name'].replace('_', ' ').title() for result in all_results]
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Customize colors
    colors = ['#ff9999', '#ff6666', '#99ccff', '#6699ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_title('API Performance - Latency Distribution for Each Test Case', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('performance_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"\nBoxplot saved as 'performance_boxplot.png'")
    
    # Create a summary table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Test Case', 'Avg (ms)', 'Min (ms)', 'Max (ms)', 'Median (ms)']]
    for result in all_results:
        table_data.append([
            result['name'].replace('_', ' ').title(),
            f"{result['avg']:.2f}",
            f"{result['min']:.2f}",
            f"{result['max']:.2f}",
            f"{result['median']:.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Performance Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    print(f"Summary table saved as 'performance_summary.png'")

def main():
    print("=" * 70)
    print("API Performance Testing")
    print("=" * 70)
    print(f"API Endpoint: {API_URL}")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"Requests per test case: 100")
    print("=" * 70)
    
    all_results = []
    
    # Run tests for each test case
    for test_name, message in test_cases.items():
        result = test_endpoint(test_name, message, num_requests=100)
        all_results.append(result)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    create_boxplots(all_results)
    
    # Print overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    overall_avg = np.mean([r['avg'] for r in all_results])
    print(f"Overall average latency across all test cases: {overall_avg:.2f} ms")
    print(f"\nAll CSV files and plots have been generated successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()


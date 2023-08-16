import googleapiclient.discovery
import os

def create_instance(
    compute: object,
    project: str,
    zone: str,
    name: str,
    bucket: str,
    accelerator_type: str,
    accelerator_count: int
) -> str:
    """Creates an instance in the specified zone.

    Args:
      compute: an initialized compute service object.
      project: the Google Cloud project ID.
      zone: the name of the zone in which the instances should be created.
      name: the name of the instance.
      bucket: the name of the bucket in which the image should be written.

    Returns:
      The instance object.
    """
    # Get the latest Debian Jessie image.
    image_response = (
        compute.images()
        .getFromFamily(project="debian-cloud", family="debian-11")
        .execute()
    )
    source_disk_image = image_response["selfLink"]

    # Configure the machine
    machine_type = "zones/%s/machineTypes/n1-standard-1" % zone
    startup_script = open(
        os.path.join(os.path.dirname(__file__), "startup-script.sh")
    ).read()
    
    config = {
        "name": name,
        "machineType": machine_type,
        # Specify the boot disk and the image to use as a source.
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": source_disk_image,
                },
            }
        ],
        # Specify a network interface with NAT to access the public
        # internet.
        "networkInterfaces": [
            {
                "network": "global/networks/default",
                "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}],
            }
        ],
        'guestAccelerators': [
            {
                'acceleratorType': f'zones/{zone}/acceleratorTypes/{accelerator_type}',
                'acceleratorCount': accelerator_count
            }
        ], 
        # Allow the instance to access cloud storage and logging.
        "serviceAccounts": [
            {
                "email": "default",
                "scopes": [
                    "https://www.googleapis.com/auth/devstorage.read_write",
                    "https://www.googleapis.com/auth/logging.write",
                ],
            }
        ],
        # Metadata is readable from the instance and allows you to
        # pass configuration from deployment scripts to instances.
        "metadata": {
            "items": [
                {
                    # Startup script is automatically executed by the
                    # instance upon startup.
                    "key": "startup-script",
                    "value": startup_script,
                },
                {"key": "bucket", "value": bucket},
            ]
        },
    }

    return compute.instances().insert(project=project, zone=zone, body=config).execute()


project = 'replit-prune'
zone = 'europe-west3-c'
compute = googleapiclient.discovery.build("compute", "v1")
instance_name = 'a100-vm'
image_family = 'debian-11'
image_project = 'debian-cloud'
machine_type = "n1-standard-4"
accelerator_type = 'nvidia-tesla-a100'
accelerator_count = 1

create_instance(
    compute = compute,
    project = project,
    zone = zone,
    name = instance_name,
    bucket = 'replit-prune',
    accelerator_type = accelerator_type,
    accelerator_count = accelerator_count
)



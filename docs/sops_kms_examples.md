SOPS + KMS Integration (AWS/GCP)

Overview
- Keep your existing PGP recipient for local development.
- Add cloud KMS recipients to decrypt in AWS/GCP using IAM/service accounts.
- No private PGP keys required in production.

AWS KMS
1) Create a symmetric CMK (KMS key) and note its ARN.
2) Grant decrypt permissions to your runtime principal (EC2 role, ECS task role, or EKS SA via IRSA).
3) Add a creation rule (or uncomment in .sops.yaml) for files such as env.json:

```yaml
creation_rules:
  - path_regex: env\.json$
    kms:
      - arn: arn:aws:kms:us-east-1:123456789012:key/abcd-efgh-...
    pgp:
      - 'CB3B963C0AB97B34BAF22D68F9D46501F1146F9F'
    encrypted_regex: '^(.*)$'
```

4) Rewrap existing files to add the new recipient:

```bash
sops updatekeys env.json
```

5) Runtime: The container entrypoint will auto-load env.json if SOPS can decrypt using instance credentials.

GCP Cloud KMS
1) Create a key ring and symmetric key (e.g., projects/PROJ/locations/global/keyRings/sops/cryptoKeys/rag-writer).
2) Grant roles/cloudkms.cryptoKeyDecrypter to your workload’s service account.
3) Add a creation rule (or uncomment in .sops.yaml):

```yaml
creation_rules:
  - path_regex: env\.json$
    gcp_kms:
      - resource_id: projects/my-proj/locations/global/keyRings/sops/cryptoKeys/rag-writer
    pgp:
      - 'CB3B963C0AB97B34BAF22D68F9D46501F1146F9F'
    encrypted_regex: '^(.*)$'
```

4) Rewrap:

```bash
sops updatekeys env.json
```

5) Runtime: SOPS uses ADC (metadata server) to decrypt; entrypoint will export env vars automatically.

Terraform Snippets

AWS (KMS + role permission)
```hcl
resource "aws_kms_key" "sops" {
  description             = "SOPS decryption key"
  deletion_window_in_days = 7
}

resource "aws_kms_alias" "sops" {
  name          = "alias/sops"
  target_key_id = aws_kms_key.sops.key_id
}

data "aws_iam_policy_document" "sops_decrypt" {
  statement {
    sid     = "AllowDecrypt"
    actions = ["kms:Decrypt", "kms:DescribeKey"]
    resources = [aws_kms_key.sops.arn]
  }
}

resource "aws_iam_policy" "sops_decrypt" {
  name   = "sops-decrypt"
  policy = data.aws_iam_policy_document.sops_decrypt.json
}

# Attach to your role (EC2/ECS/EKS)
resource "aws_iam_role_policy_attachment" "attach" {
  role       = aws_iam_role.workload.name
  policy_arn = aws_iam_policy.sops_decrypt.arn
}
```

GCP (KMS + SA binding)
```hcl
resource "google_kms_key_ring" "sops" {
  name     = "sops"
  location = "global"
  project  = var.project
}

resource "google_kms_crypto_key" "rag_writer" {
  name     = "rag-writer"
  key_ring = google_kms_key_ring.sops.id
  purpose  = "ENCRYPT_DECRYPT"
}

resource "google_kms_crypto_key_iam_binding" "decrypt" {
  crypto_key_id = google_kms_crypto_key.rag_writer.id
  role          = "roles/cloudkms.cryptoKeyDecrypter"
  members       = [
    "serviceAccount:${google_service_account.workload.email}"
  ]
}
```

Operational Notes
- Keep PGP recipient for local dev; add KMS recipients for cloud.
- Use `sops updatekeys` whenever you change recipients to re-encrypt data keys.
- Container now includes `sops` and `jq`, and auto-loads `/app/env.json` if present and decryptable.


#!/usr/bin/env php
<?php
$input = stream_get_contents(STDIN);
$data = json_decode($input, true);
$out = ["echo" => $data["message"] ?? null];
echo json_encode($out);
?>

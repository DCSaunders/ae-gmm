#!/usr/bin/perl
use strict;

sub anon_fh {
   local *FH;
   return *FH;
}

my $f_name = "/home/ds636/sentence_split";
my @buckets = (5, 10, 15);
open(my $in, "<", $f_name) 
    or die "$f_name not openable";
my %handles;

foreach $b (@buckets)
{
    $handles{$b} = anon_fh();
    open($handles{$b}, ">>",  "$f_name.$b")
	or die "$f_name.$b not openable";
}

while (<$in>)
{
    chomp;
    my $unsplit = $_;
    my @line = split(' ', $unsplit);
    foreach $b (@buckets)
    {
	if (@line < $b) {
	    print {$handles{$b}} "$unsplit\n";
	    last;
	}
    }
}

foreach (@buckets){
    close $handles{$_} or die "$handles{$_}: $!";
}

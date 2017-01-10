#!/usr/bin/perl
use strict;

sub anon_fh {
   local *FH;
   return *FH;
}

my $f_name = "/home/mifs/ds636/exps/data/wmt.en.14/test.idx";
my @buckets = (11, 21, 31, 41, 51);
open(my $in, "<", $f_name) 
    or die "$f_name not openable";
my %handles;
my $plus;
foreach $b (@buckets)
{
    $handles{$b} = anon_fh();
    open($handles{$b}, ">>",  "$f_name.$b")
	or die "$f_name.$b not openable";
}
open($plus, ">>", "$f_name.plus")
    or die "$f_name.plus not openable";

while (<$in>)
{
    chomp;
    my $unsplit = $_;
    my @line = split(' ', $unsplit);
    if (@line < 51) {
	foreach $b (@buckets)
	{
	    if (@line < $b) {
		print {$handles{$b}} "$unsplit\n";
		last;
	    }
	}
    }
    else {
	print $plus "$unsplit\n";
    }
}

foreach (@buckets){
    close $handles{$_} or die "$handles{$_}: $!";
}

close $plus or die "$plus: $!";
